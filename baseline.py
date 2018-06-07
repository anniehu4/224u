import pickle
import re
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

data = pickle.load(open('data.pkl', 'rb'), encoding='latin1')
answers = [d[0] for d in data]
scores = np.array([d[2] for d in data]).astype(np.float)
glove_home = os.path.join('vsmdata', 'glove.6B')
use_nn = True
use_embed = True

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]	

def camel_case_process(s):
	words = s.split()

	cc_split_words = []
	for w in words:
		cc_split_words += camel_case_split(w)

	joined = " ".join(cc_split_words)
	return joined

# TODO:
# separate code keywords (private, void, for, int) from non-keywords
def process(s):
	s = s.replace('(', ' ').replace(')', ' ')
	s = s.replace('\n', ' ').replace('\t', '')
	# starter code sometimes has a /** 1a **/ with the problem number
	# unclear if this is removing important comments though
	if '/**' in s:
		s = s[s.find('**/') + 3:].strip()
	# remove other punctuation (e.g. {, =, *)
	s = re.sub(r'[^\w\s]', '', s)
	# handle snake case
	s = s.replace('_', ' ')
	# remove extraneous whitespace
	s = re.sub(' +', ' ', s)
	s = camel_case_process(s)
	return s.lower()

def embed(s, lookup):
	tokens = [lookup[x] for x in s.split(' ') if x in lookup]
	return np.array(tokens)

def pad(features, max_len, dim=50):
	for i, row in enumerate(features):
		pad_size = max_len - len(row)
		if pad_size < 0:
			features[i] = row[:max_len, :].flatten()
		elif pad_size == max_len: #strange edge case, will debug later
			features[i] = np.zeros(max_len * dim)
		else:
			padded = np.pad(row, ((0, pad_size), (0, 0)), 'constant')
			features[i] = padded.flatten()
	return np.array(features)

def main():
	# TODO: better train / val / test split
	indices = list(range(len(data)))
	random.seed(1)
	random.shuffle(indices)
	split = int(0.8 * len(indices))
	trainIndices, testIndices = indices[:split], indices[split:]

	glove_lookup = utils.glove2dict(os.path.join(glove_home, 'glove.6B.50d.txt'))
	trainFeatures = []
	testFeatures = []
	print("Processing strings")
	for i in trainIndices:
		features = process(answers[i])
		if use_embed:
			features = embed(features, glove_lookup)
		trainFeatures.append(features)
	for i in testIndices:
		features = process(answers[i])
		if use_embed:
			features = embed(features, glove_lookup)
		testFeatures.append(features)

	print("Transforming answers")
	# maximum length of answer in dataset is 652
	max_len = 500
	if use_embed:
		trainX = pad(trainFeatures, max_len)
		testX = pad(testFeatures, max_len)
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