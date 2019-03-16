import torch
import _pickle as pickle
import sys
import numpy as np
import spacy


# Globals for import elsewhere
SPECIAL_CHARACTERS = {"<EOL>", "<eos>", "#", "<EOT>", "</s>", "<P>"}


def init_nlp_model(special_chars=SPECIAL_CHARACTERS,
                   model_name="en_core_web_lg"):
    """inits a spacy model and adds custom special chars to tokenizer. Returns model"""
    nlp = spacy.load(model_name)
    for key in special_chars:
        nlp.tokenizer.add_special_case(key, [dict(ORTH=key)])

    return nlp

def cosine_sim(x_d, vec_array, topN=1):
    #take dot product of 2 vectors. which reduces dimensionality and gives an array of results.
    #IMPORTANT that vec_array is first arg as a result
    dot_prod_array = np.dot(vec_array, x_d)
    len_vec_array, len_x_d = (vec_array**2).sum(axis=1)**.5, (x_d**2).sum()**.5
    cosine_sim_array = np.divide(dot_prod_array, len_vec_array*len_x_d)
    if topN == 1:
        best_vec_index = np.argmax(cosine_sim_array)
        return [(best_vec_index, cosine_sim_array[best_vec_index])]
    else:
        # TODO test this
        indices = np.argpartition(cosine_sim_array, -topN)[-topN:]
        winning_tuples = [(i, cosine_sim_array[i]) for i in indices]
        return winning_tuples


def read_w2v(w2v_path, word2index, n_dims=300, unk_token="unk"):
    """takes tokens from files and returns word vectors
    :param w2v_path: path to pretrained embedding file
    :param word2index: Counter of tokens from processed files
    :param n_dims: embedding dimensions
    :param unk_token: this is the unk token for glove 840B 300d. Ideally we make this less hardcode-y
    :return numpy array of word vectors
    """
    print('Getting Word Vectors...', file=sys.stderr)
    vocab = set()
    # hacky thing to deal with making sure to incorporate unk tokens in the form they are in for a given embedding type
    if unk_token not in word2index:
        word2index[unk_token] = 0 # hardcoded, this would be better if it was a method of a class

    word_vectors = np.zeros((len(word2index), n_dims))  # length of vocab x embedding dimensions
    with open(w2v_path) as file:
        lc = 0
        for line in file:
            lc += 1
            line = line.strip()
            if line:
                row = line.split()
                token = row[0]
                if token in word2index or token == unk_token:
                    vocab.add(token)
                    try:
                        vec_data = [float(x) for x in row[1:]]
                        word_vectors[word2index[token]] = np.asarray(vec_data)
                        if lc == 1:
                            if len(vec_data) != n_dims:
                                raise RuntimeError("wrong number of dimensions")
                    except:
                        print('Error on line {}'.format(lc), file=sys.stderr)
                    # puts data for a given embedding at an index based on the word2index dict
                    # end up with a matrix of the entire vocab
    tokens_without_embeddings = set(word2index) - vocab
    print('Word Vectors ready!', file=sys.stderr)
    print('{} tokens from text ({:.2f}%) have no embeddings'.format(
        len(tokens_without_embeddings), len(tokens_without_embeddings)*100/len(word2index)), file=sys.stderr)
    print('Tokens without embeddings: {}'.format(tokens_without_embeddings), file=sys.stderr)
    print('Setting those tokens to unk embedding', file=sys.stderr)
    for token in tokens_without_embeddings:
        word_vectors[word2index[token]] = word_vectors[word2index[unk_token]]
    return word_vectors


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def load_pickle(path):
    with open(path, 'rb') as fin:
        obj = pickle.load(fin)
    return obj

def load_model(path, cuda):
    with open(path, 'rb') as f:
        #model = torch.load(f, map_location=lambda storage, loc: storage)
        model, criterion, optimizer = torch.load(f, map_location=lambda storage, loc: storage)
    model.eval()
    if not hasattr(model, 'tie_weights'):
        model.tie_weights = True
    if cuda:
        model.cuda()
    else:
        model.cpu()
    return model


def make_vocab(corpus_dictionary, vocab_path):
    """take data, create pickle of vocabulary"""
    with open(vocab_path, 'wb') as fout:
        pickle.dump(corpus_dictionary, fout)
    print('Saved dictionary to', vocab_path)


def truncate_word_vec(file, vocab_file, unk_token='unk'):
    vocab = load_pickle(vocab_file)
    all_word_vectors = read_w2v(file, vocab.word2idx, unk_token=unk_token)
    with open(file+".vocab", "w") as outfile:
        print("Writing {} word vectors to new shortened file".format(len(vocab), file=sys.stderr))
        for word in vocab.word2idx:
            outfile.write("{} {}\n".format(word, " ".join([str(x) for x in all_word_vectors[vocab.word2idx[word]]])))
        print("Done", file=sys.stderr)







