import sys
import numpy as np
import itertools
from itertools import filterfalse
from math import log
from scipy.special import expit
import torch
from torch import nn, optim
from torch.autograd import Variable

from .candidate import Candidate
from .StaticCoefficientModel import StaticCoefficientModel


def bool_partition(func, iterable):
    """takes a function that return a bool and and iterable returns two generators from iterable, the true and the false"""
    #true_things, false_things = tee(iterable)
    return list(filter(func, iterable)), list(filterfalse(func, iterable))

def normalize_score(score, len_y, alpha=0.6):
    """takes a score, a length, and an alpha and normalizes the score and returns.
    Based on Wu et al. (2016)"""
    norm_factor = ((5 + len_y) ** alpha) / ((5 + 1) ** alpha)
    return score/norm_factor


def concat_hidden(beam, nlayers, m_per_layer=2):
    """
    takes a beam of Candidates and a number of layers and makes new concatenated layers for batching efficiency
    :param beam:
    :param nlayers:
    :param m_per_layer: matrices per later. Defaults to 2 for LSTMs
    :return: list of tuples, one tuple per layer, that are concatenation of all layers belonging
    to candidates
    """
    new_hidden = []
    for l in range(nlayers):
        # need to add an additional dimension before concatenation else get (1, 7500) instead of (1, 5, 1500) with beam of 5 and hidden layers 1500
        new_layer = tuple([torch.cat([cand.hidden[l][i].unsqueeze(1) for cand in beam], dim=1)
                           for i in range(m_per_layer)])
        new_hidden.append(new_layer)
    return new_hidden

def logprobs(model, seqs, use_cuda=True):
    hidden = model.init_hidden(len(seqs)) # init hidden to length of iterable, which should be the init and cont token integers
    if use_cuda:
        source = Variable(torch.LongTensor(seqs).t().cuda())
    else:
        source = Variable(torch.LongTensor(seqs).t())
    output, hidden = model(source, hidden) # forward
    decoded_data = model.decoder(output.data)
    output = nn.functional.log_softmax(decoded_data, dim=decoded_data.dim() - 1).data #take softmax along the final dimension
    #print(output.shape)
    return output

class BeamDecoder:
    """Upper lever class for Beamsearch and Beamrank decoders"""

    def __init__(self, model, beam_size, verbosity=0,
                 dictionary=None, temperature=None, max_len=1000, sep=None):
        self.model = model
        self.beam_size = beam_size
        self.sep = sep  # only necessary for using only one mode in decode. separates sentences or phrases etc
        self.temperature = temperature
        self.max_len = max_len  # alternate criteria for terminating generation
        self.verbosity = verbosity
        self.dictionary = dictionary

    def top_k_next(self, beam, k, temperature=None):
        """
        takes a current beam, and returns the next expansion of the beam
        :param beam: a list of Candidate objects encoding the beam sequence so far
        :param k: the k number of candidates for this expansion
        :param temp: the temp to use in softmax. Only valid if sampling.
        :return: a list lists of Candidates after the expansion (where the outer list corresponds to
        the starting candidates that were expanded and the inner to their expansions)
        """
        # cuda check
        use_cuda = next(self.model.parameters()).is_cuda

        assert (len(beam) > 0)
        first_pass = False  # used later to deal with the first beam expansion differently

        with torch.no_grad():
            if beam[0].hidden is None:
                # beam is a nested list, if this is the first pass it will be a nested list of one element
                hidden = self.model.init_hidden(len(beam))
                tokens = [cand.tokens for cand in beam]
                first_pass = True
            else:
                # this is making a tuple of tensors for each layer in hidden to track the LSTM matrices.
                # Shape is list of tuples of tensors. Used for efficiency in batching the forward function
                hidden = concat_hidden(beam, self.model.nlayers)
                assert(len(hidden) == self.model.nlayers)
                tokens = [[cand.next_token] for cand in beam]

            # tokens are a list of the next token from the previous step, coindexed with hidden layers
            if use_cuda:
                source = Variable(torch.LongTensor(tokens).t().cuda())
            else:
                source = Variable(torch.LongTensor(tokens).t())

            output, hidden = self.model(source, hidden)  # calls forward pass, returns a tensor and a list of tuples of tensors for LSTM
            decoded_data = self.model.decoder(output.data)
            ps = nn.functional.log_softmax(decoded_data,
                               dim=decoded_data.dim() - 1).data  # gives logprobs based on softmax across last dimension. Means that each slice along this dimension sums to one.

        #if not temperature:
        _, idxs = ps.topk(k)  # returns tuple of top-k values and top-k indices of the softmax transformed layers
            #print(idxs)
        #else:
            #word_weights = decoded_data.squeeze().data.div(temperature).exp().cpu()
            #idxs = torch.multinomial(word_weights, k)

        idxs_np = idxs.cpu().numpy()  # get numpy array of topk indices

        if first_pass:
            # need to select the last row, since this corresponds to the expansions of the final word in the input (and we don't want to try to expand the others)
            # also need to insert a dummy dimension so that indexing later works (since expect a 2D option as later we will be expanding more than one Candidate per search)
            ps = ps[-1, :].unsqueeze(0)
            idxs_np = np.expand_dims(idxs_np[-1], axis=0)
            #assert(12 in idxs_np)  # this is the beginning token TODO make this a real assert
        #print(idxs_np)
        beam_cands = []
        for i in range(len(beam)):  # iterate across all live paths in beam
            ith_cands = []
            base_score = beam[i].score
            # get corresponding hidden of the candidate in question after the transformation. This should basically be undoing the concatenation from earlier. The slicing must be along a column and the layers look like (1, beam size, num hidden), hence [:, i, :] for a slice
            cur_hidden = [ (hidden[l][0][:, i, :].clone(), hidden[l][1][:, i, :].clone()) for l in range(self.model.nlayers) ]
            for j in range(k):  # iterate over all possible expansions of beam for the current path. for each expansion they are in sorted order (as per pytorch topk) but the cumulative scores may not be sorted.
                next_word = int(idxs_np[i, j])
                nu_score = base_score + ps[i, next_word] #increment score of entire path
                # this is normalization by length as per Wu et al. (2016)
                #norm_score = normalize_score(nu_score, len(beam[i].tokens)+1)

                nu_cand = Candidate(beam[i].tokens + [next_word],
                                beam[i].cont_tokens + [next_word],
                                next_word,
                                score=nu_score, #norm_score
                                # TODO make sure this is also modified by normalization later in rerank since latest score is only touched by scorers
                                latest_score=beam[i].latest_score,  # I find it confusing that score is the most up to date and latest score is not. And why we need latest score. But maybe it makes more sense with the scorers...
                                hidden=cur_hidden)
                ith_cands.append(nu_cand)
            beam_cands.append(ith_cands)
        return beam_cands  # return full set of candidates. This will be a list of lists of candidates (k*k).


class BeamSearchDecoder(BeamDecoder):

    def __init__(self, model, beam_size, end_tok, verbosity=0,
                 dictionary=None, temperature=None, max_len=1000, sep=None):

        super().__init__(model, beam_size, verbosity, dictionary, temperature, max_len, sep)
        self.end_tok = end_tok  # used to knowing when to terminate generation

    def decode(self, tokens, temperature=None, keep_end=False, only_one=False):
        """
        :param tokens: list of ints corresponding to vocab words
        :param temperature: softmax temp
        :param keep_end: controls whether to pop off final token or not
        :param only_one: whether to generate only one (sentence, word, etc) based on a delimiter
        :return list of ints corresponding to vocab words - either at max length or ending with
        end token
        """
        # Validation checks
        if temperature is None:
            temperature = self.temperature
        if only_one:
            assert self.sep is not None, "Need to provide a sep token in decoder init in order to use only one mode"
        end_tok = {self.sep, self.end_tok} if only_one else {self.end_tok}

        beam = [ Candidate(tokens, []) ]
        beam = self.top_k_next(beam, self.beam_size, temperature)
        beam = beam[0]  # since initialising this basically unpacks the nested list
        #best = beam[0]  # arbitrary but should be fine
        step = 0
        final_candidates = []
        while step < self.max_len and beam and len(final_candidates) <= self.beam_size:
            #print(beam[0].convert())
            conts = self.top_k_next(beam, self.beam_size, temperature)
            beam = sorted([candidate for candidates in conts for candidate in candidates],
                          key=lambda c: c.score,
                          reverse=True)
            if not temperature:
                if len(beam) > self.beam_size:
                    beam = beam[:self.beam_size]
            else:
                if len(beam) > self.beam_size:
                    p = np.asarray(list(map(lambda c: c.score, beam)))
                    p = np.exp(p / temperature)
                    p /= p.sum()
                    beam = np.random.choice(beam, size=self.beam_size, replace=True, p=p)
            #if best is None or beam[0].score > best.score: #pick best of continuations since sorted
            has_end_tok, lacks_end_tok = bool_partition(lambda cand: cand.tokens[-1] in end_tok, beam)
            final_candidates.extend(has_end_tok)
            beam = lacks_end_tok
            step += 1
        if not final_candidates:
            print('None of candidates had end token: {}. Picking best available'.format(end_tok), file=sys.stderr)
            best = max(beam, key=lambda c: c.score)
        else:
            best = max(final_candidates, key=lambda c: c.score)  # TODO this might be unnecessary if partition is in place
        if not keep_end:
            if best.tokens[-1] in end_tok:
                best.tokens.pop()
        if self.verbosity:
            for cand in final_candidates:
                print("Score: {} \n Text: {}".format(cand.score,
                                       " ".join([self.dict.idx2word[token]
                                                              for token in cand.tokens])))
        return best.tokens



class BeamRerankDecoder(BeamDecoder):

    def __init__(self, model, scorers, coefs,
                 learn=False, lr=0.01, rescale_scores=True,
                 ranking_loss=False,
                 paragraph_level_score=False,
                 beam_size=32, terms=[1], temperature=None,
                 verbosity=0, dictionary=None,
                 max_len=150, forbidden=[], sep=1, use_cuda=True):
        super().__init__(model, beam_size, verbosity, dictionary, temperature, max_len, sep)

        self.scorers = scorers
        self.coefs = np.asarray(coefs)
        self.rescale_scores = rescale_scores
        self.terms = set(terms)
        self.learn = learn
        self.total_loss, self.total_n, self.total_correct = 0, 0, 0  # this is for learning
        self.forbidden = set(forbidden)
        self.use_ranking_loss = ranking_loss
        self.paragraph_level_score = paragraph_level_score
        self.use_cuda = use_cuda

        if self.learn:
            self.weight_model = StaticCoefficientModel(len(scorers))
            #if self.use_cuda:
            #    self.weight_model.cuda() #TODO using cuda breaks StaticCoefficientModel somewhere in the Linear forward function
            if ranking_loss:
                self.loss = nn.MarginRankingLoss()
            else:
                self.loss = nn.MSELoss()
            self.optimizer = optim.SGD(self.weight_model.parameters(), lr=lr)

    def decode(self, init_tokens, cont_tokens=None, temperature=None, rescore_min=1,
               min_sentences=5, only_one=False, keep_end=False):
        """
        :param init_tokens: ints corresponding to vocab
        :param cont_tokens: ints corresponding to gold continuation, required if learning
        :param temperature: affects broadness of search
        :param rescore_min: the minimum sentences for generate before applying rescore
        :param min_sentences: the minimum sentences to generate before stopping
        :param only_one: if True, makes the seperator token also an end token.
        :param keep_end: controls whether to pop off final token or not
        :return: if not in learn mode, the beam sequence tokens. If in learn mode, the diff score between...
        """
        ### Validation checks
        assert((not self.learn) or cont_tokens)
        if temperature is None:
            temperature = self.temperature # TODO this is not ideal since it forces using beamrank with temp unless the decoder was initialised with None
        if self.learn:
            self.coefs = self.weight_model.coefs.weight.data.cpu().squeeze().numpy()

        end_terms = self.terms
        if only_one:
            end_terms.add(self.sep)
            min_sentences = 1

        beam = [ Candidate(init_tokens, []) ]
        beam = self.top_k_next(beam, self.beam_size, temperature)[0]  # picks first of the k returned as part of init. Since this is the first expansion, it is presorted
        beam = list(filter(lambda c: c.cont_tokens[-1] not in self.forbidden, beam)) # filter out options where final continuation token is in the list of forbidden (usually unk)
        sentences_count = 1
        gold_cont_raw_scores, best = None, None
        step = 2  # used for learning below and also to control max iterations. But why must it start at 2?
        cont_latest_scores = log(0.34) #TODO WHY hardcoding log(0.34)? Something to do with 3 options...same thing is hardcoded in Candidate, but also this is only used once to append to cand_latest_scores before it has been set so index 0 will be a bogus value?
        while (((best is None) or (best.adjusted_score < max(map(lambda c: c.score, beam)))) and (step < self.max_len)):  # max len is 150 seemingly arbitrarily. So that it can't beam search forever
            rescore = True if (len(self.scorers) and sentences_count > rescore_min) else False # whether to rescore

            if self.verbosity > 0:
                print("rescore: ", rescore)
                for c in beam:
                    print(' '.join([self.dictionary[i] for i in c.cont_tokens]) + ' %f' % c.score)
                print('-'*30)

            #get topk next
            conts = self.top_k_next(beam, self.beam_size, temperature)

            if self.verbosity > 0:
                for cs in conts:
                    for c in cs:
                        print(' '.join([self.dictionary[i] for i in c.cont_tokens]) + ' %f' % c.score)
                print('*'*50)
                if self.verbosity > 2:
                    input()

            candidates, cand_cont_tokens, cand_latest_scores = [], [], []
            for cands in conts:
                for candidate in cands:
                    candidates.append(candidate)
                    cand_cont_tokens.append(candidate.cont_tokens)  # this will append all continuation tokens of each candidate
                    cand_latest_scores.append(candidate.latest_score)  # this is always log(0.34 on init)

            if self.learn and step < len(cont_tokens):  # add gold answer to the list
                cand_cont_tokens.append(cont_tokens[:step]) # this appends all gold cont tokens up to the step number - so more gold on each iteration
                cand_latest_scores.append(cont_latest_scores)  # the gold answer have a high probability, here it is hardcoded as 0.34.

            # score adjustment section
            score_adjustment = np.zeros(len(candidates)) # since this is redone on each while loop - candidates is len k*k
            if rescore:  # add score adjustment according to the scorers.
                all_raw_scores = []
                for coef, scorer in zip(self.coefs, self.scorers): # Note that this throws an error if there is just one scorer
                    # this makes an array for each scorer from calling the scorer forward function on the candidate tokens
                    new_scores = scorer(init_tokens, cand_cont_tokens,  #rescale scores, if True, causes the scores to be normalized. Paragraph level scores seems unused.
                        cand_latest_scores, self.terms, self.rescale_scores,
                        self.paragraph_level_score)  # TODO the scores that are rescaled aren't the beamsearch scores, they are the latest scores (hardcoded)
                    raw_scores = np.asarray(new_scores)
                    #print(len(raw_scores), len(candidates))
                    all_raw_scores.append(raw_scores)
                    # elementwise add the new scores to the np array after elementwise multiplying by coef
                    score_adjustment += raw_scores[:len(candidates)] * coef  # TODO why restrict to len(candidates)? It seems like the scorer sometimes but not always returns +1 more result than candidates
                last_raw_scores = all_raw_scores[-1] # all_raw scores will be num_scorers x num_candidates. So last_raw_scores is just the last scorer results?
                all_raw_scores = np.stack(all_raw_scores, axis=-1)  # this converts to num_candidates x num_scorers so each row is all adjusted scores for a candidate

                if self.learn and step < len(cont_tokens):
                    gold_cont_raw_scores = all_raw_scores[-1]  # this is the adjusted scores for the gold, since it was appended last
                    cont_latest_scores = gold_cont_raw_scores[-1] # this will be the specific score for the most recently added continuation token

            # score adjustments are zero if no scorers. Basically this enable them to use candidate.adjusted_score regardless if scorers are present
            for i, candidate in enumerate(candidates):
                candidate.adjusted_score = candidate.score + score_adjustment[i]
                if rescore:
                    candidate.latest_score = last_raw_scores[i] # this is the only place where latest score is modified. TODO this seems like a bug...or at least like an unnecessary line.
                    candidate.raw_scores = all_raw_scores[i]  # this is the candidate's scores from the scorers (arrray)

            candidates = sorted(candidates, key=lambda c: c.adjusted_score, reverse=True)
            filtered_candidates = list(filter(lambda c: c.cont_tokens[-1] not in self.forbidden, candidates))

            if temperature and len(filtered_candidates) > self.beam_size:
                p = np.asarray(list(map(lambda c: c.adjusted_score, filtered_candidates)))
                p = np.exp(p / temperature)
                p /= p.sum()
                beam = np.random.choice(filtered_candidates, size=self.beam_size, replace=True, p=p)  #TODO replace=True? why?
            # since candidates is sorted this just prunes the beam
            else:
                beam = [cand for cand in itertools.islice(filter(lambda c: c.cont_tokens[-1] not in self.forbidden, candidates), self.beam_size)]

            for candidate in filter(lambda c: c.cont_tokens.count(self.sep) == min_sentences and c.cont_tokens[-1] in end_terms, candidates): # 1 is the index of the sentence continuation token, terms is ending terms.
                if best is None or candidate.adjusted_score > best.adjusted_score:
                    best = candidate
            sentences_count = max(map(lambda c: c.cont_tokens.count(self.sep), candidates)) # used for seeing how many have been generated
            step += 1
        best = best or beam[0]

        if self.learn:
            self.weight_model.zero_grad()
            truth_lm_scores = logprobs(self.model, [init_tokens + cont_tokens], use_cuda=self.use_cuda).squeeze().cpu().data.numpy() # this will be the shape of (len input x embed dimension) where len input is init + cont
            truth_lm_score = sum([truth_lm_scores[i+len(init_tokens)-1, cont_tokens[i]] for i in range(len(cont_tokens))]) #this is just the probability of the sequence
            lm_scores = torch.Tensor([truth_lm_score, beam[0].score])/50  # this is the probability of the true sequence paired with the score of the best sequence. Both floats
            #print("LM pair", lm_scores)
            training_pair = [gold_cont_raw_scores, beam[0].raw_scores] # this is scorer scores of gold continuation, and of the best continuation. Both 1D arrays of len num scorers.
            training_pair = torch.Tensor(np.stack(training_pair)) # so this is now one row per scorer, with gold and best candidate as columns
            #print("Training pair", training_pair)
            #if self.use_cuda:
            #    training_pair.cuda()
            pair_scores = self.weight_model(training_pair).squeeze()
            #print("pair scores returned", pair_scores)
            pair_scores = pair_scores + lm_scores
            #print("pair scores concat", pair_scores)
            pred = pair_scores[0] - pair_scores[1]

            if self.use_ranking_loss:
              loss = self.loss((pair_scores[0]).unsqueeze(0),
                               (pair_scores[1]).unsqueeze(0), Variable(torch.ones(1)))

            else:
              loss = self.loss(pred, torch.FloatTensor([0]))  # use MSELoss, ((input-target)**2).mean()
            #print(loss.data.item())
            loss.backward()
            self.total_loss += loss.data.item()
            if self.use_ranking_loss and loss.data.item() == 0:
                self.total_correct += 1 # whether or not it is correct is whether the scorer did in fact say the gold was higher rank
            self.total_n += 1
            if self.total_n % 200 == 0:
                if self.use_ranking_loss:
                    print('Train Accuracy: %f' % (self.total_correct / self.total_n))
                print('Loss: %f' % (self.total_loss/ self.total_n))
                sys.stdout.flush()

            self.optimizer.step()
            self.weight_model.coefs.weight.data = self.weight_model.coefs.weight.data.clamp(min=0)

        if not keep_end:
            if best.tokens[-1]in self.terms:  # avoid printing end_tok
                best.tokens.pop()
        return best.tokens if not self.learn else loss

