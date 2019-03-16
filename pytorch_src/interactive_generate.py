"""
Allows interactive generation (without many bells and whistles) via the command line

Example args (many more args are available, but have defaults)
--keyword-vocab
../models/ROC_title_keyword_e500_h1000_edr0.4_hdr0.1_511_lr10.pkl
--keyword-model
../models/ROC_title_keyword_e500_h1000_edr0.4_hdr0.1_511_lr10.pt
--story-vocab
../models/ROC_title_key_story_e1000_h1500_edr0.2_hdr0.1_511_lr10.pkl
--story-model
../models/ROC_title_key_story_e1000_h1500_edr0.2_hdr0.1_511_lr10.pt
--oov
wordnet
--collaborate-kw
--collaborate-st

"""

#TODO add a y/n/stop to allow user to terminate generation before the model wants to

import argparse
import sys
import os
import random
from collections import defaultdict, deque
from importlib import import_module
import torch
from nltk.corpus import wordnet
import spacy
import numpy as np
from utils import batchify, get_batch, repackage_hidden, load_pickle, load_model, read_w2v, cosine_sim, init_nlp_model
from decoder.decoders import BeamRerankDecoder, BeamSearchDecoder


def read_scorers(config, cuda):
    """reads a tsv for scorer config and returns scorer models and their weight coefficients"""
    scorer_config, scorers, coefs = [], [], []
    with open(config) as scorer_file:
        for line in scorer_file:
            fields = line.strip().split('\t')
            scorer_config.append(fields)
            weight, module_path, classname = fields[:3]
            weight = float(weight)
            model_location = fields[3]
            module = import_module(module_path)
            constructor = getattr(module, classname)
            scorer = constructor(model_location,
                                 use_cuda=cuda)  # this calls a scorer init, but a scorer init expects a pretrained model
            scorers.append(scorer)
            coefs.append(weight)
    return scorers, coefs

def read_gold_storylines(gold_data_path, eot_char):
    title2storyline = defaultdict(list) # init as empty
    if gold_data_path is None or len(gold_data_path) == 0:
        return title2storyline
    with open(gold_data_path, 'r') as fin:
        for line in fin:
            title, storyline = line.strip().split(eot_char) # after this the title will not have an eot char but the storyline will. This is fine downstream.
            title, storyline = title.strip(), storyline.strip()
            title2storyline[title].append(storyline)
    return title2storyline

def to_tensor(vocab, tokens, cuda):
    """takes a covab word2idx dict and a list of string and returns a tensor of ints"""
    ids = torch.LongTensor(len(tokens))
    for i, token in enumerate(tokens):
        ids[i] = vocab.word2idx.get(token, 0)
    if cuda:
        ids = ids.cuda()
    return ids


def generate(model, vocab, prefix, eos_id, delimiter_idx, max_len, dedup, cuda, temperature,
             only_one=False, forbidden=None):
    """takes a model, vocab, and a list of vocab integers representing the conditional input
    only_one is used to generate only one keyword/keyword phrase at a time. Forbidden prevents certain keywords from being generated
    returns a lists of string tokens """
    forbidden = set() if not forbidden else set(forbidden)
    cond_length = len(prefix)
    hidden = model.init_hidden(1)
    tokens = []
    input = torch.rand(1, 1).mul(max_len).long()
    if cuda:
        input.data = input.data.cuda()
    for i in range(max_len):
        if i < cond_length:
            word_idx = prefix[i]
        else:
            output = model.decoder(output)
            word_weights = output.squeeze().data.div(temperature)
            word_weights = (word_weights - torch.max(word_weights)).exp()#.cpu()
            samples = torch.multinomial(word_weights, 5)
            for word_idx in samples:
                word_idx = word_idx.item()
                if word_idx not in forbidden or word_idx == delimiter_idx:
                    break
            if dedup:
                forbidden.add(word_idx)

            word = vocab.idx2word[word_idx]
            tokens.append(word)
            if word_idx == eos_id or (word_idx == delimiter_idx and only_one):
                break
        input.data.fill_(word_idx)
        output, hidden = model(input, hidden)

    return tokens


def preprocess(word_string, special_chars, tokenizer):
    """
    Takes a word string and a set of special characters and returns a processed word list

    tokenizes input in case it was user generated.
    A bit tricky because tokenization won't help if the algo used for it doesn't match whatever
    tokenized the training data. Uses spacy because the tokenizer is easiest to customize.
    """
    # currently converting this to take and return a list. If a word split by whitespace
    #ended up as multiple tokens then list will have to be flattened. Google it.
    # Then afterwards make sure that it plays nice with main function
    tok_text = tokenizer(word_string)
    return [word.text.strip().lower() if word.text.strip() not in special_chars else word.text.strip() for word in tok_text]


def to_ints(word_list, vocab):
    """takes a list of strings, returns ints based on vocab"""
    return [vocab.word2idx.get(w, 0) for w in word_list]


def to_str(int_list, vocab):
    """takes list of ints and vocab, returns a string"""
    return ' '.join([vocab.idx2word[i] for i in int_list])


def check_for_oov(word_list, word2idx):
    oov_words = list(filter(lambda w: w not in word2idx, word_list))
    if oov_words:
        return True
    else:
        return False


def find_wordnet_substitute(word, word2idx, max_depth=10):
    """takes a word and a word2idx dict and tries to find substitutes using wordnet
    :param word: a string
    :param word2idx: a word2idx dict
    :param max_depth: the maximum distance wordnet can go from the original word up or down tree
    :return better word (string) if found, else None
    """
    all_senses = wordnet.synsets(word)
    if not all_senses:
        print("wordnet does not appear to know about the word: {}".format(word), file=sys.stderr)
        return None
    best_senses = deque([all_senses[0]]) # picks the most common one, which is how wordnet orders.
    best_sense = best_senses.pop()
    init_depth = best_sense.max_depth()
    # first try to get lemmas and see if any of those are in vocab
    while best_sense not in word2idx and abs(best_sense.max_depth() - init_depth) <= max_depth:
        lemmas = list(filter(lambda l: l in word2idx, best_sense.lemma_names()))  # will always filter out words with an underscore since it represents whitespace and we don't have internal whitespace
        if lemmas:
            one_lemma = random.choice(lemmas)
            dict_def = wordnet.synsets(one_lemma)[0].definition()
            print("picking word: {} with definition: {}".format(one_lemma, dict_def), file=sys.stderr)
            return one_lemma
        else:
            best_senses.extend(best_sense.hypernyms())
            best_senses.extend(best_sense.hyponyms())
        if len(best_senses) >= 1: # it is possible for a word to have no hyper or hyponyms
            best_sense = best_senses.popleft()
        else:
            break
    print("Got to {} depth on word {} before failing".format(best_sense.max_depth(), best_sense.name()), file=sys.stderr)


def find_vector_substitute(word, vocab, vocab_word_vec, nlp, threshold=0.25):
    """takes a word and a vocab dict and tries to find substitutes using preloaded vectors
    :param word: a string
    :param vocab: a vocab object with word2idx, idx2word etc
    :param vocab_word_vec: the premade word vectors for the vocabulary
    :param nlp: a larger set of word vectors that may contain the oov vector
    :return better word (string) if found, else None. Technically could always find one, but currently limit on similarity threshold.
    """
    vector_dim = len(vocab_word_vec[0])
    this_vec = nlp(word)
    if this_vec.has_vector:
        this_vec = this_vec.vector
    else:
        print("No vector found for {}, cannot find similar words", file=sys.stderr)
        return None
    assert(len(this_vec) == vector_dim), "preloaded vector dim needs to match other vector dim"
    best_indices = cosine_sim(this_vec, word_vectors, topN=10)
    closest_idx = None
    for idx, sim in best_indices:
        if sim > 0.99:
            continue
        else:
            closest_idx, best_sim = idx, sim
    print("Most similar word was {} with sim {}".format(vocab.idx2word[closest_idx], best_sim),
          file=sys.stderr)
    if best_sim > threshold:
        return vocab.idx2word[closest_idx]
    else:
        print("best sim is below threshold of {}".format(threshold), file=sys.stderr)
        return None


def replace_oovs(word_list, vocab, oov_handling, vocab_vectors=None, all_vectors=None ):
    """
    :param words: a string
    :param vocab: a dictionary object with word2idx, idx2word etc dicts
    :param oov_handling: a string for the type of oov handling to apply
    :param special_chars: a set of special characters. Empty if None
    :param word_vectors: necessary if not using wordnet (to replace words)
    :return: a string with similar words replacing the original oov ones
    """
    if not check_for_oov(word_list, vocab):
        return word_list
    #TODO move oov handling into a dict of string to function so cleaner. But might need globals...
    for i in range(len(word_list)):
        if word_list[i] in vocab:
            continue
        if oov_handling == "wordnet":
            new_word = find_wordnet_substitute(word_list[i], vocab)
        elif oov_handling == "wordvec":
            assert(len(vocab_vectors) > 0), "Need to provide preloaded word vectors to use wordvec for OOV"
            new_word = find_vector_substitute(word_list[i], vocab, vocab_vectors, all_vectors)
        else:
            print("invalid choice for oov: {}, returning original string".format(oov_handling), file=sys.stderr)
            return word_list

        if new_word: #if something was returned
            print("subbing {} instead of original {}".format(new_word, word_list[i]), file=sys.stderr)
            word_list[i] = new_word
        else:
            print("couldn't find a substitute for {}".format(word_list[i]), file=sys.stderr)

    return word_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Language Model')

    # Model parameters.
    parser.add_argument('--keyword-vocab', type=str, required=True, default=None)
    parser.add_argument('--story-vocab', type=str, required=True, default=None)
    parser.add_argument('--keyword-model', type=str, required=True)
    parser.add_argument('--story-model', type=str, required=True)
    parser.add_argument('--gold-storylines', type=str, default=None,
                        help='a file with titles and storyline to use as for random gen and to retrieve gold storylines')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')

    ### DECODING PARAMS ###
    parser.add_argument('--words', type=int, default='1000',
                        help='number of words to generate')
    parser.add_argument('--kw-temp', type=float, default=0.5,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--story-temp', type=float, default=None,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--beam-size', type=int, default=10, help="size of beam for beam search (k)")
    parser.add_argument('--apply-disc', action='store_true',
                        help='whether to apply discriminators to output')
    parser.add_argument('--scorers', type=str, default=None,
                        help='tsv with scorer information')
    parser.add_argument('--dedup', action='store_true',
                        help='de-duplication, generally used for storyline keywords')

    ### Formatting and Interaction ###
    parser.add_argument('--print-cond-data', action='store_true',
                        help='whether to print conditional data used for generation')
    parser.add_argument('--collaborate-kw', action='store_true',
                        help='generates one storyline keyword continuation at a time and asks user to accept or edit')
    parser.add_argument('--collaborate-st', action='store_true',
                        help='generates one story continuation at a time and asks user to accept or edit')
    parser.add_argument('--oov', type=str,
                        help='options: wordvec, wordnet. Type of oov handling to employ for user input titles')
    parser.add_argument('--word-vec', type=str, help='word vectors file if using for oov handling')
    ### Default delimiters and special symbols  ###
    parser.add_argument('--kw-end', type=str, default='<EOL>',
                        help='what string to use as the end token')
    parser.add_argument('--kw-sep', type=str, default='#',
                        help='what string to use to separate storyline keywords')
    parser.add_argument('--story-end', type=str, default='<eos>',
                        help='what string to use as the end token')
    parser.add_argument('--story-sep', type=str, default='</s>',
                        help='what string to use to separate story sentences')

    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    # Load models and vocab dictionaries, init stopping symbols for generation
    kw_model = load_model(args.keyword_model, args.cuda)
    st_model = load_model(args.story_model, args.cuda)
    kw_dict = load_pickle(args.keyword_vocab)
    st_dict = load_pickle(args.story_vocab)
    kw_vocab_size = len(kw_dict)
    st_vocab_size = len(st_dict)
    st_eos_id = st_dict.word2idx[args.story_end]
    eot_char = '<EOT>'
    special_chars = {args.kw_end, args.story_end, args.kw_sep, args.story_sep, eot_char}
    title2storyline = defaultdict(list) # init as empty

    # Init vectors for OOV handling
    word_vectors = None
    if args.oov == "wordvec":
        assert args.word_vec, "Need to provide a file of word vectors to use this option"
        # read word vectors into numpy array. Ideally file already truncated to vocab size for speed
        print("Loading word vectors for oov handling", file=sys.stderr)
        word_vectors = read_w2v(args.word_vec, kw_dict.word2idx)  # NOTE that the coindexing is for kw_dict so this cannot be used for stories
        nlp = init_nlp_model(special_chars=special_chars, model_name="en_vectors_web_lg")
    else:
    # Need to Init a spacy model even if not using it for OOV handling, since have to tokenize user input
        nlp = init_nlp_model(special_chars=special_chars)


    # Init Decoders for Story
    if args.apply_disc:
        print('Init discriminators', file=sys.stderr)
        assert(args.scorers), "Need to provide a tsv file for scorer config"
        scorers, coefs = read_scorers(args.scorers, args.cuda)

        decoder = BeamRerankDecoder(st_model,
                                   scorers,
                                   coefs,
                                   beam_size=args.beam_size,
                                   temperature=args.story_temp,
                                   sep=st_dict.word2idx[args.story_sep],
                                   terms=[st_eos_id],
                                   dictionary=st_dict.idx2word,
                                   forbidden=[st_dict.word2idx['<unk>']],
                                   use_cuda=args.cuda,
                                   #verbosity=True
                                   )
    else:
        decoder = BeamSearchDecoder(st_model, args.beam_size, st_eos_id, verbosity=False,
                                    sep=st_dict.word2idx[args.story_sep], dictionary=st_dict, temperature=args.story_temp)
    if args.gold_storylines:
        print("Reading gold title to storyline data", file=sys.stderr)
        title2storyline = read_gold_storylines(args.gold_storylines, eot_char)

    # TODO there is some messiness here in that the generate function returns strings and the decode returns vocab ints. Should standardise.
    while True:
        orig_title = input("Enter a Title:  ") #  TODO support a random choice title via args.gold_storylines
        tok_title = preprocess(orig_title, special_chars, nlp)
        # if any OOV words exist, use something else. If none found it will just return the input.
        if args.oov == "wordnet":
            mod_title = replace_oovs(tok_title, kw_dict, args.oov)
        elif args.oov == "wordvec":
            mod_title = replace_oovs(tok_title, kw_dict, args.oov, word_vectors, nlp)
        else:
            mod_title = tok_title
        title = to_ints(mod_title, kw_dict) + [kw_dict.word2idx[eot_char]]  # title ends with special char

        if args.collaborate_kw:
            print('Model generating keywords one at a time: ')
            all_tokens = title
            used_word_ints = set()
            while all_tokens[-1] != kw_dict.word2idx[args.kw_end]:
                next_kw = generate(kw_model, kw_dict, all_tokens, kw_dict.word2idx[args.kw_end],
                                   kw_dict.word2idx[args.kw_sep], 25, args.dedup, args.cuda, args.kw_temp, only_one=True, forbidden=used_word_ints)
                # TODO also need to make sure if the user edits the EOL that they don't make it keep going. Or maybe that is desirable?
                print(' '.join(next_kw))
                user_kw_choice = input('Accept generated keyword (y/n)?: ')
                use_model_keywords = True if user_kw_choice == 'y' else False
                if not use_model_keywords:
                    user_kw = input('Enter a keyword or keyword phrase, ending with a # '
                                    'to keep generating more phrases or <EOL> if it is the last phrase: ')
                    next_kw = preprocess(user_kw, special_chars, nlp)
                    print('using "{}" as the next keyword'.format(next_kw))
                kw_ints = to_ints(next_kw, kw_dict)
                all_tokens.extend(kw_ints)
                if args.dedup: # add keywords to forbidden. Note that a user can still add a duplicate keyword when args.dedup is true.
                    used_word_ints.update([idx for idx in kw_ints
                                           if idx != kw_dict.word2idx[args.kw_sep]])  # the delimiter can't be forbidden as the generate function exempts it, but still good to do here.
            all_tokens = to_str(all_tokens, kw_dict) # have to convert to string before passing to story generation since the word2idx mapping is different for the different models (different vocabs)
            print('\nFull sequence:\n{}\n'.format(all_tokens))

        else: # Don't collaborate, but allow user to accept or reject
            # if the title exists in our training set, use gold standard keywords
            if args.gold_storylines and (mod_title in title2storyline):
                print('Retrieved storyline:')
                # title storyline mappings are not guaranteed to be unique. Currently picks first, could code differently
                kw_cont_string = title2storyline[mod_title][0]
            else:
                print('Model generated storyline: ')
                # Storyline ends with EOL
                kw_cont_tokens = generate(kw_model, kw_dict, title, kw_dict.word2idx[args.kw_end],
                              kw_dict.word2idx[args.kw_sep], 25, args.dedup, args.cuda, args.kw_temp)
                kw_cont_string = ' '.join(kw_cont_tokens)

            print(kw_cont_string)

            user_kw_choice = input('Use model storyline (y/n): ')
            use_model_keywords = True if user_kw_choice == 'y' else False
            if not use_model_keywords:
                user_kw = input('Enter your own storyline (# to separate phrases, ending with <EOL>): ')
                kw_cont_string = " ".join(preprocess(user_kw, special_chars, nlp))
            all_tokens = to_str(title, kw_dict) + ' ' + kw_cont_string

        story_prefix = to_ints(all_tokens.split(), st_dict)  # Toggling back and forth between ints and strings is necessary since different models have different vocab mappings
        prompt_len = len(story_prefix) # store here because prefix length will be modified if collaborating

        with torch.no_grad():
            if args.collaborate_st:
                print("Collaborating on story generation...\n")

                while story_prefix[-1] != st_dict.word2idx[args.story_end]:
                    new_story_prefix = decoder.decode(story_prefix, only_one=True, keep_end=True)
                    cont_tokens = [st_dict.idx2word[new_story_prefix[j]] for j in
                                   range(len(story_prefix), len(new_story_prefix))]
                    print("Model Generated Sentence: {}".format(' '.join(cont_tokens)))
                    user_sent_choice = input('Accept generated sentence (y/n)?: ')
                    use_model_sent = True if user_sent_choice == 'y' else False
                    if not use_model_sent:
                        user_sent = input('Enter a sentence, ending with a </s>: ')
                        print('using "{}" as the next sentence'.format(user_sent))
                        next_sent = preprocess(user_sent, special_chars, nlp)
                        new_sent_ints = to_ints(next_sent, st_dict)
                        story_prefix.extend(new_sent_ints)
                    else:
                        story_prefix = new_story_prefix
                st_tokens = story_prefix
            else:
                st_tokens = decoder.decode(story_prefix)

        cont_tokens = to_str(st_tokens[prompt_len:], st_dict)
        if args.print_cond_data:
            init_tokens = to_str(st_tokens[:prompt_len], st_dict)
            cont_tokens = init_tokens + cont_tokens
        print("\nFull output:\n{}\n".format(cont_tokens))
