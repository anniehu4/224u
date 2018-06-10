import pickle
import random
import numpy as np
import time
import os
import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pytorch_model import *
import matplotlib.pyplot as plt
import keyword
from utils import *
from vocab import Vocabulary

vocab = pickle.load(open('vocab.pkl', 'rb'))
data = pickle.load(open('data.pkl', 'rb'))
answers = [d['answer'] for d in data]
scores = np.array([d['score'] for d in data]).astype(np.float)
glove_home = os.path.join('vsmdata', 'glove.6B')
use_nn = True
use_rnn = True
use_embed = True

def main():
	# TODO: better train / val / test split
	indices = list(range(len(data)))
	random.seed(1)
	random.shuffle(indices)
	split = int(0.8 * len(indices))
	trainIndices, testIndices = indices[:split], indices[split:]

	glove_lookup = glove2dict(os.path.join(glove_home, 'glove.6B.50d.txt'))
	trainFeatures = []
	testFeatures = []
	print("Processing strings")
	for i in trainIndices:
		features = filter_keywords(process(answers[i]))
		if use_embed: # TODO: make embedding compatible with rnn
			features = embed(features, glove_lookup)
		elif use_rnn:
			features = [vocab(x) for x in features.split(' ')]
		trainFeatures.append(features)
	for i in testIndices:
		features = filter_keywords(process(answers[i]))
		if use_embed:
			features = embed(features, glove_lookup)
		elif use_rnn:
			features = [vocab(x) for x in features.split(' ')]
		testFeatures.append(features)

	print("Transforming answers")
	# maximum length of answer in dataset is 652
	max_len = 500
	if use_embed:
		trainX, lengths = pad(trainFeatures, max_len)
		testX, lengths = pad(testFeatures, max_len)
	else:
		cv = CountVectorizer()
		trainX = cv.fit_transform(trainFeatures).toarray()
		testX = cv.transform(testFeatures).toarray()
	print("Train size: {} Test size: {}".format(trainX.shape, testX.shape))
	trainY, testY = scores[trainIndices], scores[testIndices]

	print("Training")
	if use_nn:
		train_true, train_pred, test_true, test_pred = dnn(trainX, trainY.reshape(-1, 1), testX, testY.reshape(-1, 1))
	else:
		model = LinearRegression(fit_intercept=False)
		model.fit(trainX, trainY)
		predictions = model.predict(trainX)
		predictions[predictions > 30] = 30
		predictions[predictions < 0] = 0
		print("Train R2: {}".format(r2_score(trainY, predictions)))

		predictions = model.predict(testX)
		predictions[predictions > 30] = 30
		predictions[predictions < 0] = 0
		print("Test R2: {}".format(r2_score(testY, predictions)))
	
	plt.scatter(test_true, test_pred)
	plt.show()



if __name__ == '__main__':
	main()