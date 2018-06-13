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

arg_parser = argparse.ArgumentParser(description="parser for cs224u project")
arg_parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")

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
arg_parser.add_argument('--classify', action='store_true', default=False,
                        help='classification problem if True, regression otherwise')
arg_parser.add_argument('--use-spellcheck', action='store_true', default=False,
                        help='in preproc, spellcheck all words if true')
arg_parser.add_argument('--normalize-scores', action='store_true', default=True,
                        help='True to predict normalized scores (min 0, max 1)')
arg_parser.add_argument('--glove-dim', type=int, default=200,
						help='dimension for GloVe embeddings')
args = arg_parser.parse_args()

holdout = True 
TRAIN_FILE = 'data/%strain.pkl' % 'holdout-' if holdout else ''
DEV_FILE = 'data/%sdev.pkl' % 'holdout-' if holdout else ''


def main():
	print("Loading training data...")
	train_data = pickle.load(open(TRAIN_FILE, 'rb'))
	dev_data = pickle.load(open(DEV_FILE, 'rb'))
	vocab = pickle.load(open('data/vocab.pkl', 'rb'))

	if args.classify:
		args.normalize_scores = True

	# answers is list of strings, scores is numpy array of shape (len,)
	train_answers, train_scores = prepare_data(train_data, args.normalize_scores)
	dev_answers, dev_scores = prepare_data(dev_data, args.normalize_scores)
	print(" - done.")

	if args.use_embed:
		print("Preparing glove data...")
		glove_home = os.path.join('vsmdata', 'glove.6B')
		glove_lookup = glove2dict(os.path.join(glove_home, 'glove.6B.%dd.txt' % args.glove_dim))
		print(" - done.")

	train_x = []
	dev_x = []

	print("Processing strings")
	print("Using model: {}".format(args.model))
	print("Using embeddings: {}".format(args.use_embed))
	if args.model == "nn":
		print("Using collate function: {}".format(args.collate_fn))
	for train_answer in train_answers:
		features = process(train_answer, glove_lookup, args.remove_numbers, args.use_spellcheck)
		# RNN should have data as timeseries
		if args.model == "rnn":
			# with embeddings, each timestep is a glove vector of shape=(args.glove_dim, 1)
			if args.use_embed:
				features = embed(features, glove_lookup, dim=args.glove_dim)
			# without embeddings, each time step is a word index of shape=(1,)
			else:
				features = [vocab(x) for x in features.split(' ')]	
		elif args.model == "nn":
			features = embed(features, glove_lookup, dim=args.glove_dim, collate_fn=args.collate_fn)
		else:
			raise NotImplementedError
		if not isinstance(features, float):
			train_x.append(features)

	for dev_answer in dev_answers:
		features = process(dev_answer, glove_lookup, args.remove_numbers, args.use_spellcheck)
		# RNN should have data as timeseries
		if args.model == "rnn":
			# with embeddings, each timestep is a glove vector of shape=(args.glove_dim, 1)
			if args.use_embed:
				features = embed(features, glove_lookup, dim=args.glove_dim)
			# without embeddings, each time step is a word index of shape=(1,)
			else:
				features = [vocab(x) for x in features.split(' ')]	
		elif args.model == "nn":
			features = embed(features, glove_lookup, dim=args.glove_dim, collate_fn=args.collate_fn)
		else:
			raise NotImplementedError
		if not isinstance(features, float):
			dev_x.append(features)
	print(" - done.")

	print("Transforming answers")
	# maximum length of answer in dataset is 652
	"""
	if args.use_embed:
		max_len = max([len(x) for x in train_x])
		if not args.collate:
			train_x = train_x, max_len)
			dev_x = pad(dev_x, max_len)
	elif not args.model == "rnn":
		cv = CountVectorizer()
		train_x = cv.fit_transform(train_x).toarray()
		dev_x = cv.transform(dev_x).toarray()
	"""
	print("Train size: {} Dev size: {}, # dimensions: {}".format(
		len(train_x), len(dev_x), max([len(x) for x in train_x])))
	train_y, dev_y = train_scores, dev_scores
	if args.classify:
		threshold = 1.0
		train_y = np.array([1.0 if x >= threshold else 0.0 for x in train_scores])
		dev_y = np.array([1.0 if x >= threshold else 0.0 for x in dev_scores])

	print("Training")
	if args.model == "nn":

		train_true, train_pred, dev_true, dev_pred = basic_nn(np.array(train_x), train_y.reshape(-1, 1), np.array(dev_x), dev_y.reshape(-1, 1), args.classify)
	elif args.model == "rnn":
		train_true, train_pred, dev_true, dev_pred = rnn(train_x, train_y, dev_x, dev_y)
	else:
		model = LinearRegression(fit_intercept=False)
		model.fit(train_x, train_y)
		train_pred = model.predict(train_x)
		train_pred[train_pred > 30] = 30
		train_pred[train_pred < 0] = 0
		print("Train R2: {}".format(r2_score(train_y, train_pred)))

		dev_pred = model.predict(dev_x)
		dev_pred[dev_pred > 30] = 30
		dev_pred[dev_pred < 0] = 0
		dev_true = dev_y
		print("Dev R2: {}".format(r2_score(dev_y, dev_pred)))
	
	plt.scatter(dev_true, dev_pred)
	plt.show()



if __name__ == '__main__':
	main()