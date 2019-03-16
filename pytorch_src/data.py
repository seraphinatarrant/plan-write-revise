import os
import torch

from collections import Counter

from utils import load_pickle, make_vocab


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0


    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, applyDict=False, **kwargs):
        """
        :param applyDict: whether to create a corpus with an already made dictionary
        :param kwargs: 'train_path' 'dev_path' 'test_path', 'dict_path', 'output'. For most uses
        you need all types of path, though you could make a Corpus without a train-dev-test split.
        dict_path is only accessed if applyDict is true.
        """
        if applyDict:
            self.dictionary = load_pickle(kwargs['dict_path'])  # a previously saved pickle of a Dictionary
        else:
            self.dictionary = Dictionary()
            if 'train_path' in kwargs.keys():
                self.train = self.tokenize(kwargs['train_path'])
            if 'dev_path' in kwargs.keys():
                self.valid = self.tokenize(kwargs['dev_path'])
            if 'test_path' in kwargs.keys():
                self.test = self.tokenize(kwargs['test_path'])
            # save file when done
            make_vocab(self.dictionary, kwargs['output'])


    def tokenize(self, path, applyDict=False):
        """Tokenizes a text file."""
        assert os.path.exists(path), path
        # Add words to the dictionary
        tokens = 0
        if not applyDict:
            with open(path, 'r') as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)
        # Tokenize file content
        if not tokens:
            with open(path, 'r') as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)  # init the LongTensor to size of tokens, pretty large and 1D
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx.get(word, 0)
                    token += 1

        return ids
