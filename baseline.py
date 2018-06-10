import pickle
import random
import numpy as np
import time
import os
import sys
import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pytorch_model import *
import matplotlib.pyplot as plt
from utils import *
from vocab import Vocabulary

def main():
	print("Loading training data...")
	train_data = pickle.load(open('data/train.pkl', 'rb'))
	dev_data = pickle.load(open('data/dev.pkl', 'rb'))
	vocab = pickle.load(open('data/vocab.pkl', 'rb'))

	normalize_scores = False
	use_nn = False
	use_rnn = True
	use_embed = False

	train_answers, train_scores = prepare_data(train_data, normalize_scores)
	dev_answers, dev_scores = prepare_data(dev_data, normalize_scores)
	print(" - done.")

	print("Preparing glove data...")
	glove_home = os.path.join('vsmdata', 'glove.6B')
	# glove_lookup = glove2dict(os.path.join(glove_home, 'glove.6B.50d.txt'))
	glove_lookup = None
	print(" - done.")
	train_x = []
	dev_x = []

	print("Processing strings")
	for train_answer in train_answers:
		features = filter_keywords(process(train_answer))
		if use_embed:
			features = embed(features, glove_lookup)
		elif use_rnn:
			features = [vocab(x) for x in features.split(' ')]
		train_x.append(features)
	for dev_answer in dev_answers:
		features = filter_keywords(process(dev_answer))
		if use_embed:
			features = embed(features, glove_lookup)
		elif use_rnn:
			features = [vocab(x) for x in features.split(' ')]
		dev_x.append(features)
	print(" - done.")

	print("Transforming answers")
	# maximum length of answer in dataset is 652
	max_len = 500
	if use_embed:
		train_x, lengths = pad(train_x, max_len)
		dev_x, lengths = pad(dev_x, max_len)
	elif not use_rnn:
		cv = CountVectorizer()
		train_x = cv.fit_transform(train_x).toarray()
		dev_x = cv.transform(dev_x).toarray()
	print("Train size: {} Dev size: {}, # dimensions: {}".format(
		len(train_x), len(dev_x), max([len(x) for x in train_x])))
	train_y, dev_y = train_scores, dev_scores

	print("Training")
	if use_nn:
		train_true, train_pred, dev_true, dev_pred = basic_nn(train_x, train_y.reshape(-1, 1), dev_x, dev_y.reshape(-1, 1))
	elif use_rnn:
		train_true, train_pred, dev_true, dev_pred = rnn(train_x, train_y.reshape(-1, 1), dev_x, dev_y.reshape(-1, 1))
	else:
		model = LinearRegression(fit_intercept=False)
		model.fit(train_x, train_y)
		predictions = model.predict(train_x)
		predictions[predictions > 30] = 30
		predictions[predictions < 0] = 0
		print("Train R2: {}".format(r2_score(train_y, predictions)))

		predictions = model.predict(dev_x)
		predictions[predictions > 30] = 30
		predictions[predictions < 0] = 0
		print("Dev R2: {}".format(r2_score(dev_y, predictions)))
	
	plt.scatter(dev_true, dev_pred)
	plt.show()



if __name__ == '__main__':
	main()