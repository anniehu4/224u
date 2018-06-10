import pickle
import re
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
import keyword

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

def filter_keywords(s):
	words = []
	keywords = []
	count_keywords = 0 # just to sanity check
	for x in s.split(' '):
		if keyword.iskeyword(x):
			keywords.append(x)
			count_keywords += 1
		else:
			words.append(x)
	return ' '.join(words)

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

def prepare_data(data, use_normalized):
	answers = [d['answer'] for d in data]
	if use_normalized:
		scores = np.array([d['scoreNormalized'] for d in data]).astype(np.float)
	else:
		scores = np.array([d['score'] for d in data]).astype(np.float)

	return (answers, scores)



def main():
	print("Loading training data...")
	train_data = pickle.load(open('data/train.pkl', 'rb'))
	dev_data = pickle.load(open('data/dev.pkl', 'rb'))

	normalize_scores = False
	use_nn = True
	use_embed = False

	train_answers, train_scores = prepare_data(train_data, normalize_scores)
	dev_answers, dev_scores = prepare_data(dev_data, normalize_scores)
	print(" - done.")

	print("Preparing glove data...")
	glove_home = os.path.join('vsmdata', 'glove.6B')

	glove_lookup = utils.glove2dict(os.path.join(glove_home, 'glove.6B.50d.txt'))
	print(" - done.")
	trainFeatures = []
	devFeatures = []

	print("Processing strings")
	for train_answer in train_answers:
		features = filter_keywords(process(train_answer))
		if use_embed:
			features = embed(features, glove_lookup)
		trainFeatures.append(features)
	for dev_answer in dev_answers:
		features = filter_keywords(process(dev_answer))
		if use_embed:
			features = embed(features, glove_lookup)
		devFeatures.append(features)
	print(" - done.")

	print("Transforming answers")
	# maximum length of answer in dataset is 652
	max_len = 500
	if use_embed:
		trainX = pad(trainFeatures, max_len)
		devX = pad(devFeatures, max_len)
	else:
		cv = CountVectorizer()
		trainX = cv.fit_transform(trainFeatures).toarray()
		devX = cv.transform(devFeatures).toarray()
	print("Train size: {} Dev size: {}".format(trainX.shape, devX.shape))
	trainY, devY = train_scores, dev_scores

	print("Training")
	if use_nn:
		train_true, train_pred, dev_true, dev_pred = dnn(trainX, trainY.reshape(-1, 1), devX, devY.reshape(-1, 1))
	else:
		model = LinearRegression(fit_intercept=False)
		model.fit(trainX, trainY)
		predictions = model.predict(trainX)
		predictions[predictions > 30] = 30
		predictions[predictions < 0] = 0
		print("Train R2: {}".format(r2_score(trainY, predictions)))

		predictions = model.predict(devX)
		predictions[predictions > 30] = 30
		predictions[predictions < 0] = 0
		print("Dev R2: {}".format(r2_score(devY, predictions)))
	
	plt.scatter(dev_true, dev_pred)
	plt.show()



if __name__ == '__main__':
	main()