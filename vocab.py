import nltk
import pickle
import argparse
import re
from collections import Counter
from utils import process

data_path = 'data/data.pkl'
vocab_output_path = 'data/vocab.pkl'

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(pkl, threshold=0):
    """Build a simple vocabulary wrapper."""
    data = pickle.load(open(pkl, 'rb'))
    answers = [d['answer'] for d in data]
    counter = Counter()
    for answer in answers:
        tokens = process(answer).split(' ')
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    print(vocab.word2idx)
    return vocab

def main():
    vocab = build_vocab(pkl=data_path, threshold=0.0)
    with open(vocab_output_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_output_path))


if __name__ == '__main__':
    main()