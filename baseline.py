import pickle
import re
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

data = pickle.load(open('data.pkl', 'rb'))
answers = [d[0] for d in data]
scores = np.array([d[1] for d in data])

def process(s):
	s = s.replace('(', ' ').replace(')', ' ')
	s = s.replace('\n', ' ').replace('\t', '')
	if '/**' in s:
		s = s[s.find('**/') + 3:].strip()
	s = re.sub(r'[^\w\s]', '', s)
	s = re.sub(' +', ' ', s)
	return s


def main():
	indices = range(len(data))
	random.shuffle(indices)
	split = int(0.8 * len(indices))
	trainIndices, testIndices = indices[:split], indices[split:]
	trainFeatures = []
	testFeatures = []
	for i in trainIndices:
		trainFeatures.append(process(answers[i]))
	for i in testIndices:
		testFeatures.append(process(answers[i]))
	print "Transforming answers"
	cv = CountVectorizer()
	trainX = cv.fit_transform(trainFeatures)
	vocab = cv.vocabulary_
	testX = cv.transform(testFeatures)
	print "Train size:", trainX.shape, "Test size:", testX.shape
	trainY, testY = scores[trainIndices], scores[testIndices]
	print "Training"
	model = LinearRegression()
	model.fit(trainX, trainY)
	print "Train R2:", model.score(trainX, trainY)
	print "Test R2:", model.score(testX, testY)


if __name__ == '__main__':
	main()