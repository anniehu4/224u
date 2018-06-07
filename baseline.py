import pickle
import re
import random
import numpy as np
import os
import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data = pickle.load(open('data.pkl', 'rb'), encoding='latin-1')
answers = [d[0] for d in data]
scores = np.array([d[2] for d in data])
glove_home = os.path.join('vsmdata', 'glove.6B')

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
		trainFeatures.append(embed(process(answers[i]), glove_lookup))
	for i in testIndices:
		testFeatures.append(embed(process(answers[i]), glove_lookup))
	print("Transforming answers")
	cv = CountVectorizer()
	trainX = cv.fit_transform(trainFeatures)
	testX = cv.transform(testFeatures)
	print("Train size: {} Test size: {}".format(trainX.shape, testX.shape))
	trainY, testY = scores[trainIndices], scores[testIndices]

	print("Training")
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
	# plt.scatter(testY, predictions)
	# plt.show()



if __name__ == '__main__':
	main()