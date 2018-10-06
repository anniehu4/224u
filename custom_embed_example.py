import nltk
import pickle
import argparse
import re
from collections import Counter
from gensim import models

caption_path = 'combined/trainImages/labels.pkl'
vocab_output_path = 'combined/train_vocab.pkl'
embedding_output_path = 'combined/train_embedding.bin'

IGNORE = ['\\mbox', '\\left', '\\right', '\\hbox', '\\vtop']

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

def tokenize(caption):
    cleaned_caption = caption.replace('{', ' ')
    cleaned_caption = cleaned_caption.replace('}', ' ')
    tokens = []
    for word in cleaned_caption.split(' '):
        # commands like \sqrt need to not be split
        in_command = False
        token = ''
        for ch in list(word):
            if ch == '\\':
                in_command = True
                token = ch
            elif (not ch.isalpha() and not ch.isdigit()) or (
                in_command and ch.isdigit()):
                in_command = False
                if token and token not in IGNORE:
                    tokens.append(token)
                token = ch
            elif in_command:
                token += ch
            else:
                token = ch

            if not in_command and token and token not in IGNORE:
                tokens.append(token)
                token = ''
        if token and token not in IGNORE:
            tokens.append(token)

    return tokens

def build_embeddings(pkl, vocab):
    """Simple Word2Vec embeddings"""
    captions = pickle.load(open(pkl, 'rb'))
    corpus = [['<pad>'], ['<start>'], ['<end>'], ['<unk>']]
    for i, im_caption in enumerate(captions):
        caption = im_caption[1]
        corpus.append(tokenize(caption))

    # corpus = [[vocab(token) for token in tokens] for tokens in corpus]
    model = models.Word2Vec(corpus, size=100, window=5, min_count=1)

    return model

def build_vocab(pkl, threshold=0):
    """Build a simple vocabulary wrapper."""
    captions = pickle.load(open(pkl, 'rb'))
    counter = Counter()
    for i, im_caption in enumerate(captions):
        caption = im_caption[1]
        tokens = tokenize(caption)
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    print(vocab.word2idx)
    return vocab

def main():
    vocab = build_vocab(pkl=caption_path, threshold=0.0)
    with open(vocab_output_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_output_path))

    embeddings = build_embeddings(caption_path, vocab)
    embeddings.save(embedding_output_path)
    print("Total embedding size: {}".format(len(embeddings.wv.vocab)))
    print("Saved embeddings to '{}'".format(embedding_output_path))


if __name__ == '__main__':
    main()