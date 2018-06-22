import pickle
import argparse
import random
import numpy as np
import time
import os
import sys
import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from pytorch_model import basic_nn, rnn
import matplotlib.pyplot as plt
from utils import *
from vocab import Vocabulary
from split_data import holdout_split

arg_parser = argparse.ArgumentParser(description="parser for cs224u project")
arg_parser.add_argument('--data_dir', default='processed/', help="Directory containing the dataset")

arg_parser.add_argument("--name", type=str, default=None,
                              help="name for the model")
arg_parser.add_argument('--model', type=str, default="nn",
                              help="nn or rnn")
arg_parser.add_argument('--use-embed', action='store_true', default=True,
                        help='True to use GloVe embeddings')
arg_parser.add_argument('--collate-fn', type=str, default="sum",
                        help='avg or sum, used for nn model only')
arg_parser.add_argument('--remove-numbers', action='store_true', default=False,
                        help='in preproc, replace numbers if true')
arg_parser.add_argument('--strip-starter', action='store_true', default=True,
                        help='in prepare data, strip starter code if true')
arg_parser.add_argument('--classify', action='store_true', default=False,
                        help='classification problem if True, regression otherwise')
arg_parser.add_argument('--use-spellcheck', action='store_true', default=False,
                        help='in preproc, spellcheck all words if true')
arg_parser.add_argument('--normalize-scores', action='store_true', default=True,
                        help='True to predict normalized scores (min 0, max 1)')
arg_parser.add_argument('--glove-dim', type=int, default=200,
						help='dimension for GloVe embeddings')
arg_parser.add_argument('--pretrained', action='store_true', default=False,
                        help='True to use pretrained model checkpoint')
args = arg_parser.parse_args()

# test on held-out question
holdout = True
TRAIN_FILE = args.data_dir + '%strain.pkl' % ('holdout-' if holdout else '')
DEV_FILE = args.data_dir + '%sdev.pkl' % ('holdout-' if holdout else '')
TEST_FILE = args.data_dir + '%stest.pkl' % ('holdout-' if holdout else '')
# from FinalWin18, can be changed
PROBLEMS = ['1a', '1b', '1c', '1d', '2', '3a', '3b', '4', '5', '6', '7']

def main():
	print("Loading training data...")
	train_data = pickle.load(open(TRAIN_FILE, 'rb'))
	if args.pretrained:
		dev_data = pickle.load(open(TEST_FILE, 'rb'))
	else:
		dev_data = pickle.load(open(DEV_FILE, 'rb'))
	vocab = pickle.load(open('data/vocab.pkl', 'rb'))

	total = 0.0
	best = float("-inf")
	best_q = ''
	best_devx = None
	best_devy = None

	if args.classify:
		args.normalize_scores = True
	if args.use_embed:
		print("Preparing glove data...")
		glove_lookup = glove2dict('vsmdata/glove.6B/glove.6B.%dd.txt' % args.glove_dim)
		print(" - done.")

	for i in range(len(PROBLEMS)):
		prob = PROBLEMS[i]
		train_data, dev_data, _ = holdout_split(prob, prob)

		# answers is list of strings, scores is numpy array of shape (len,)
		train_answers, train_scores = prepare_data(train_data, args.normalize_scores, args.strip_starter)
		dev_answers, dev_scores = prepare_data(dev_data, args.normalize_scores, args.strip_starter)
		print(" - done.")

		train_x = []
		dev_x = []

		print("Processing strings")
		print("Using model: {}".format(args.model))
		print("Using embeddings: {}".format(args.use_embed))
		if args.model == "nn":
			print("Using collate function: {}".format(args.collate_fn))
		for train_answer in train_answers:
			# if data is already processed + spellchecked:
			# features = train_answer
			# otherwise:
			features = process(train_answer, glove_lookup, args.remove_numbers, args.use_spellcheck)
			# RNN should have data as timeseries
			if args.model == "rnn":
				# with embeddings, each timestep is a glove vector of shape=(args.glove_dim, 1)
				if args.use_embed:
					features = embed(features, glove_lookup)
				# without embeddings, each time step is a word index of shape=(1,)
				else:
					features = [vocab(x) for x in features.split(' ')]	
			elif args.model == "nn":
				features = embed(features, glove_lookup, collate_fn=args.collate_fn)
			else:
				raise NotImplementedError
			if not isinstance(features, float):
				train_x.append(features)

		for dev_answer in dev_answers:
			# if data is already processed + spellchecked:
			# features = dev_answer
			# otherwise:
			features = process(dev_answer, glove_lookup, args.remove_numbers, args.use_spellcheck)
			# RNN should have data as timeseries
			if args.model == "rnn":
				# with embeddings, each timestep is a glove vector of shape=(args.glove_dim, 1)
				if args.use_embed:
					features = embed(features, glove_lookup)
				# without embeddings, each time step is a word index of shape=(1,)
				else:
					features = [vocab(x) for x in features.split(' ')]	
			elif args.model == "nn":
				features = embed(features, glove_lookup, collate_fn=args.collate_fn)
			else:
				raise NotImplementedError
			if not isinstance(features, float):
				dev_x.append(features)
		print(" - done.")
		print("Train size: {} Dev size: {}, # dimensions: {}".format(
			len(train_x), len(dev_x), max([len(x) for x in train_x])))

		train_y, dev_y = train_scores, dev_scores
		if args.classify:
			threshold = 1.0
			train_y = np.array([1.0 if x >= threshold else 0.0 for x in train_scores])
			dev_y = np.array([1.0 if x >= threshold else 0.0 for x in dev_scores])

		print("Training")
		if args.model == "nn":

			train_true, train_pred, dev_true, dev_pred, metric = basic_nn(np.array(train_x), train_y.reshape(-1, 1),
				np.array(dev_x), dev_y.reshape(-1, 1), args.pretrained, args.classify)
		elif args.model == "rnn":
			train_true, train_pred, dev_true, dev_pred, metric = rnn(train_x, train_y, dev_x, dev_y, args.pretrained, args.classify)
		
		total += metric
		if metric > best:
			best = metric
			best_devx = dev_true
			best_devy = dev_pred

	print('best:', best, 'average:', total / len(PROBLEMS), 'best question:', best_q)
	plt.scatter(best_devx, best_devy)
	plt.show()



if __name__ == '__main__':
	main()
