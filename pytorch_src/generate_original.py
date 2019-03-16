###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import sys
import numpy, math
import torch
import torch.nn as nn
from numbers import Number
from utils import batchify, get_batch, repackage_hidden
from ir_baseline import IRetriver

import data

parser = argparse.ArgumentParser(description='PyTorch Language Model')

# Model parameters.
parser.add_argument('--train-data', type=str, default='data/penn/train.txt',
                    help='location of the training data corpus. Used for the rescore_story function')
parser.add_argument('--vocab', type=str, default='../models/vocab.pickle',
                    help='path to a pickle of the vocab used in training the model')
parser.add_argument('--keywords', type=str, default='',
                    help='location of the file for validation keywords')
parser.add_argument('--conditional-data', type=str, default='',
                    help='location of the file that contains the content that the generation conditions on')
parser.add_argument('--happy-endings', type=str, default='',
                    help='location of the file for all happy endings')
parser.add_argument('--sad-endings', type=str, default='',
                    help='location of the file for all sad endings')
parser.add_argument('--story-body', type=str, default='',
                    help='location of the file for story body')
parser.add_argument('--true-endings', type=str, default='',
                    help='location of the file for true endings')
parser.add_argument('--fake-endings', type=str, default='',
                    help='location of the file for fake endings')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--task', type=str, default='generate',
                    choices=['generate', 'cond_generate', 'shannon_game', 'rescore_ending', 'rescore_story', 'scoring'],
                    help='specify the generation task')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--sents', type=int, default='40',
                    help='number of sentences to generate')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--dedup', action='store_true',
                    help='de-duplication')
parser.add_argument('--print-cond-data', action='store_true',
                    help='whether to print the prompt on which conditionally generated text is conditioned')
parser.add_argument('--bptt', type=int, default=5000,
                    help='sequence length')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
            dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

def evaluate(data, hidden, args):
    bdata = batchify(torch.LongTensor(data), test_batch_size, args)
    #print ('current sentence: %d, ending: %d, and hidden: %s' %(j, i, str(lhidden)))
    source, targets = get_batch(bdata, 0, args, evaluation=True)
    loutput, lhidden = model(source, hidden)
    output_flat = loutput.view(-1, ntokens)
    #print ('output_flat:', output_flat.size())
    total_loss = criterion(output_flat, targets).data
    #print ('total_loss:', total_loss.size(), total_loss)
    return total_loss[0], lhidden

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    #model = torch.load(f, map_location=lambda storage, loc: storage)
    model, criterion, optimizer = torch.load(f, map_location=lambda storage, loc: storage)
model.eval()
if not hasattr(model, 'tie_weights'):
    model.tie_weights = True
if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()


corpus = data.Corpus(applyDict=True, dict_path=args.vocab)
ntokens = len(corpus.dictionary)
criterion = nn.CrossEntropyLoss()  #NLLLoss()
hidden = model.init_hidden(1)
input = torch.rand(1, 1).mul(ntokens).long()  #, volatile=True)
# For HappyEnding and SadEnding Generation
ending_dict = {0:'*HappyEnding', 1:'*SadEnding', 2:'*OtherEnding'}
#start_word = ending_dict[numpy.random.randint(0,3)]
#input.data.fill_(corpus.dictionary.word2idx[start_word])
print('ntokens', ntokens, file=sys.stderr)
#print('input:', input)
if args.cuda:
    input.data = input.data.cuda()

######### GLOBALS #########
eos_id = corpus.dictionary.word2idx['<eos>']
delimiter = '#'  # this delimits multi-word phrases. Only used to prevent the delimiter from being deduped in cond_generate when flag present
delimiter_idx = corpus.dictionary.word2idx[delimiter]
print('eos id:', eos_id, file=sys.stderr)
#data = corpus.test.tolist() #test.tolist()
#punc_idxs = set([corpus.dictionary.word2idx[p] for p in '.?!'])
test_batch_size = 1

with torch.no_grad():
  if args.task == 'rescore_story':
    assert args.keywords
    myir = IRetriver(args.keywords, args.train_data)
    count = 0
    with open(args.keywords) as kf, open(args.outf, 'w') as outf:
        for line in kf:
            count += 1
            kws = line.strip().split()
            best_story = []
            hidden = model.init_hidden(1)
            for i, kw in enumerate(kws[:5]):
                sentences = myir.query(kw)
                min_cross_ent = math.inf
                best_sent = None
                for j, sent in enumerate(sentences):
                    #print('processing keyword %d, sentence %d' %(i, j))
                    sent_idxs = [corpus.dictionary.word2idx[wd] for wd in sent]
                    if i == 0:
                        sent_idxs = [eos_id] + sent_idxs
                    else:
                        sent_idxs = best_story[-1:] + sent_idxs
                    cross_ent, _ = evaluate(sent_idxs, hidden, args)
                    if cross_ent < min_cross_ent:
                        min_cross_ent = cross_ent
                        best_sent = sent_idxs
                best_story.extend(best_sent[1:] if best_sent else [])
                if best_sent:
                    _, hidden = evaluate(best_sent, hidden, args)
            print('finished %d story!' % count)
            for word_idx in best_story:
                word = corpus.dictionary.idx2word[word_idx]
                outf.write(word + ' ')
            outf.write('\n')
            outf.flush()
            #print('best story:', [corpus.dictionary.idx2word[idx] for idx in best_story])


  elif args.task == 'rescore_ending':
    assert args.happy_endings and args.sad_endings
    happy_endings = corpus.tokenize(args.happy_endings, applyDict=True).tolist()
    sad_endings = corpus.tokenize(args.sad_endings, applyDict=True).tolist()
    endings = [happy_endings, sad_endings]
    ending_names = ['*HappyEnding ', '*SadEnding ']
    data = corpus.test.tolist() #test.tolist()
    with open(args.outf, 'w') as outf:
        for j in range(args.sents):
            idx = data.index(eos_id)
            sent_idxes = [i for i, w in enumerate(data[:idx]) if w in punc_idxs]
            cond_length = sent_idxes[-2]
            try:
                assert len(sent_idxes) == 5
            except:
                print('wrong sentence numbers:', len(sent_idxes))
            for i in range(cond_length-1):
                word_idx = data[i]
                input.data.fill_(word_idx)
                output, hidden = model(input, hidden)
            for i, x_endings in enumerate(endings):
                start_idx = -1
                end_idx = 0
                min_cross_ent = math.inf
                eidx = 0
                best_ending = None
                while start_idx+1 < len(x_endings):
                    end_idx = x_endings[start_idx+1:].index(eos_id) + (start_idx+1)
                    #print('current start and end idx:', start_idx, end_idx)
                    h_ending = [data[cond_length-1]] + x_endings[start_idx+1:end_idx]
                    #print('current ending:', len(h_ending), list(map(lambda x:corpus.dictionary.idx2word[x], h_ending)))
                    cross_ent, _ = evaluate(h_ending, hidden, args)
                    if cross_ent < min_cross_ent:
                        #print ('change the best ending!!!', eidx, cross_ent, min_cross_ent)
                        min_cross_ent = cross_ent
                        arg_max = eidx
                        best_ending = h_ending
                    #print ('the best ending:', best_ending)
                    eidx += 1
                    start_idx = end_idx

                for word_idx in data[:cond_length]:
                    word = corpus.dictionary.idx2word[word_idx]
                    outf.write(word + ' ')
                outf.write(ending_names[i])
                outf.write('\t >>\t')
                for word_idx in best_ending[1:]:
                    word = corpus.dictionary.idx2word[word_idx]
                    outf.write(word + ' ')
                outf.write('\n')
                outf.flush()
            print('finished sentence %d' % j)
            data = data[idx+1:]

  # scoring the endings according to language models.
  elif args.task == 'scoring':
    assert args.true_endings and args.fake_endings
    true_endings = corpus.tokenize(args.true_endings, applyDict=True).tolist()
    fake_endings = corpus.tokenize(args.fake_endings, applyDict=True).tolist()
    data = corpus.tokenize(args.story_body, applyDict=True).tolist() #test.tolist()
    with open(args.true_endings+'.score', 'w') as tof, open(args.fake_endings+'.score', 'w') as fof:
        j = 0
        correct, ecorrect, scorrect = 0.0, 0.0, 0.0
        while 1:
            try:
                idx = data.index(eos_id)
            except:
                break
            cond_length = idx #sent_idxes[-2]
            hidden = model.init_hidden(1)
            input.data.fill_(eos_id)
            _, hidden = model(input, hidden)
            tidx = true_endings.index(eos_id)
            fidx = fake_endings.index(eos_id)
            #print ('sentence and ending idxes:', idx, tidx, fidx)
            #print('ending idx:',tidx)
            if tidx == 1:
                true_cross_ent, true_cond_cross_ent = 0.0, 0.0
            else:
                true_cross_ent, _ = evaluate(true_endings[:tidx], hidden, args)
            if fidx == 1:
                fake_cross_ent, fake_cond_cross_ent = 0.0, 0.0
            else:
                fake_cross_ent, _ = evaluate(fake_endings[:fidx], hidden, args)
            for i in range(cond_length):
                word_idx = data[i]
                input.data.fill_(word_idx)
                _, hidden = model(input, hidden)
            assert data[i+1] == eos_id
            if tidx != 1:
                true_cond_cross_ent, _ = evaluate(true_endings[:tidx], hidden, args)
            if fidx != 1:
                fake_cond_cross_ent, _ = evaluate(fake_endings[:fidx], hidden, args)
            tscore = true_cond_cross_ent - true_cross_ent
            fscore = fake_cond_cross_ent - fake_cross_ent
            tof.write(str(true_cond_cross_ent) + '\t' + str(true_cross_ent) + '\t' + str(tscore) +'\t'+ str(tscore<fscore) + '\n')
            tof.flush()
            fof.write(str(fake_cond_cross_ent) + '\t' + str(fake_cross_ent) + '\t' + str(fscore) +'\t'+ str(fscore<tscore) + '\n')
            fof.flush()
            if tscore < fscore:
                correct += 1.0
            elif tscore == fscore:
                correct += 0.5
            if true_cross_ent < fake_cross_ent:
                ecorrect += 1.0
            elif true_cross_ent == fake_cross_ent:
                ecorrect += 0.5
            if true_cond_cross_ent < fake_cond_cross_ent:
                scorrect += 1.0
            elif true_cond_cross_ent == fake_cond_cross_ent:
                scorrect += 0.5
            data = data[idx+1:]
            true_endings = true_endings[tidx+1:]
            fake_endings = fake_endings[fidx+1:]
            j += 1
            print('finished sentence %d, scores: %f, %f.' % (j, tscore, fscore))
            print('detailed scores: %f, %f, %f, %f.' % (true_cond_cross_ent, true_cross_ent, fake_cond_cross_ent, fake_cross_ent))
    print('accuracy: '+str(correct/j)+' '+str(ecorrect/j)+' '+str(scorrect/j))

  elif args.task == 'shannon_game':
    with open('shannon_game.txt', 'w') as sf:
        for j in range(args.sents):
            idx = data.index(eos_id)
            sent_idxes = [i for i, w in enumerate(data[:idx]) if w in punc_idxs]
            cond_length = sent_idxes[-2]
            try:
                assert len(sent_idxes) == 5
            except:
                print('wrong sentence numbers:', len(sent_idxes))
            for i in range(idx):
                word_idx = data[i]
                if i > cond_length:
                    word_distr = output.squeeze().data.cpu()
                    norm = log_sum_exp(word_distr)
                    for i, w in enumerate(word_distr):
                        if i == word_idx:
                            tw = w
                        sf.write(corpus.dictionary.idx2word[i]+':'+str(w-norm)+' ')
                    sf.write('\n')
                    print(corpus.dictionary.idx2word[word_idx]+':'+str(tw-norm))
                input.data.fill_(word_idx)
                output, hidden = model(input, hidden)
            data = data[idx+1:]
            sf.write('\n')
            print('| Generated {}/{} sentences'.format(j+1, args.sents))
        sf.flush()

  else:
    with open(args.outf, 'w') as outf: #, open('gold_4sent.txt', 'w') as gf:
        if args.task == 'cond_generate':
            data = corpus.tokenize(args.conditional_data, applyDict=True).tolist()  # this is a list of ids corresponding to words from the word2idx dict
            nsent = 0
            while nsent < args.sents:
                try:
                    idx = data.index(eos_id)  # the only thing that breaks the while loop is if there are no more eos_ids
                    hidden = model.init_hidden(1)
                except:
                    break
                # this sets the conditional length to be before the first encountered EOS symbol, which is added in preprocessing
                cond_length = idx #min(idx, 3) #ent_idxes[-3] #0] #-2]
                #print('cond. length:', idx)
                exist_word = set()
                for i in range(args.words):
                    if i < cond_length:
                        word_idx = data[i]
                    else:
                        output = model.decoder(output)
                        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
                        samples = torch.multinomial(word_weights, 5)
                        #max_idx= torch.argmax(word_weights)
                        if args.dedup:
                            for word_idx in samples:
                                word_idx = word_idx.item()
                                if word_idx not in exist_word or word_idx == delimiter_idx:
                                    break
                            #print('dedup!!!', exist_word, samples, word_idx)
                            exist_word.add(word_idx)
                        else:
                            word_idx = samples[0] #max_idx
                    input.data.fill_(word_idx)
                    output, hidden = model(input, hidden)
                    if word_idx == eos_id :
                        outf.write('\n')
                        break
                    word = corpus.dictionary.idx2word[word_idx]
                    #print('word {} word_idx {}'.format(word, word_idx))
                    if i < cond_length: # prints the prompt that is conditioned on only when flag is present
                        if args.print_cond_data:
                            outf.write(word + ' ')
                    else:
                        outf.write(word + ' ')
                    #if i == cond_length-1:
                    #    outf.write('\t >>\t')
                data = data[idx+1:]  # start after the previous idx id. This is super inefficient.
                print('| Generated {} sentences'.format(nsent+1), file=sys.stderr)
                nsent += 1
            #gf.flush()
            outf.flush()
        else:
            assert args.task == 'generate'
            #outf.write(start_word + ' ')
            for i in range(args.words):
                output, hidden = model(input, hidden)
                output = model.decoder(output)
                word_weights = output.squeeze().data.div(args.temperature).exp()#.cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                #if word_idx == eos_id:
                #    start_word = ending_dict[numpy.random.randint(0,3)]
                input.data.fill_(word_idx) #if word_idx != eos_id else corpus.dictionary.word2idx[start_word])
                #print('word_idx:', word_idx)
                word = corpus.dictionary.idx2word[word_idx]

                outf.write(word + ' ') #('\n'+start_word+' ' if word_idx == eos_id else ' '))
                #outf.write(word + ('\n' if i % 20 == 19 else ' '))

                if i % args.log_interval == 0:
                    print('| Generated {}/{} words'.format(i, args.words), file=sys.stderr)
