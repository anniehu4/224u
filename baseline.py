import pickle
import argparse
import random
import numpy as np
import time
import os
import sys
import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from split_data import holdout_split

import matplotlib.pyplot as plt
from utils import *
from vocab import Vocabulary

# to test with held-out questions
holdout = True
# data directory
direc = 'processed/' 
TRAIN_FILE = direc + '%strain.pkl' % ('holdout-' if holdout else '')
DEV_FILE = direc + '%sdev.pkl' % ('holdout-' if holdout else '')
TEST_FILE = direc + '%stest.pkl' % ('holdout-' if holdout else '')
# from FinalWin18, can be changed
PROBLEMS = ['1a', '1b', '1c', '1d', '2', '3a', '3b', '4', '5', '6', '7']
classify = False

print("Loading training data...")
if not holdout:
	train_data = pickle.load(open(TRAIN_FILE, 'rb'))
	dev_data = pickle.load(open(DEV_FILE, 'rb'))

total = 0.0
best = 0
best_q = ''
best_devx = None
best_devy = None
for prob in PROBLEMS:
	if holdout:
		print("Loading problem {}".format(prob))
		train_data, dev_data, _ = holdout_split(prob, prob)

	# answers is list of strings, scores is numpy array of shape (len,)
	train_answers, train_scores = prepare_data(train_data, True, True)
	dev_answers, dev_scores = prepare_data(dev_data, True, True)
	print(" - done.")

	train_x = train_answers
	dev_x = dev_answers
	threshold = 0.5
	train_y = np.array([1.0 if x >= threshold else 0.0 for x in train_scores])
	dev_y = np.array([1.0 if x >= threshold else 0.0 for x in dev_scores])

	print("Transforming answers")	

	cv = CountVectorizer()
	train_x = cv.fit_transform(train_x).toarray()
	dev_x = cv.transform(dev_x).toarray()
	print("Train size: {} Dev size: {}".format(train_x.shape, dev_x.shape))

	# TRAIN
	model = LogisticRegression() if classify else LinearRegression(fit_intercept=False)
	model.fit(train_x, train_y)
	train_pred = model.predict(train_x)
	if classify:
		precision, recall, f1, _ = metrics.precision_recall_fscore_support(train_y, train_pred, average='binary')
		print("Train f1: {}".format(f1))
	else:
		# force into valid score range (TODO: use sigmoid instead)
		train_pred[train_pred > 1] = 1
		train_pred[train_pred < 0] = 0
		print("Train R2: {}".format(r2_score(train_y, train_pred)))

	# TEST
	dev_pred = model.predict(dev_x)
	if classify:
		precision, recall, f1, _ = metrics.precision_recall_fscore_support(dev_true, dev_pred, average='binary')
		metric = f1
		print("Dev f1: {}".format(f1))
	else:
		dev_pred[dev_pred > 1] = 1
		dev_pred[dev_pred < 0] = 0
		metric = r2_score(dev_y, dev_pred)
		print("Dev R2: {}".format(metric))

	total += metric
	if metric > best:
		best = metric
		best_q = prob
		best_devx = dev_y
		best_devy = dev_pred

print('best:', best, 'average:', total / len(PROBLEMS))
plt.scatter(best_devx, best_devy)
plt.show()