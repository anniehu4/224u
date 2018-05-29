import pickle
import re
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

data = pickle.load(open('data.pkl', 'rb'))
answers = [d["answer"] for d in data]
scores = np.array([d["score"] for d in data]).astype(np.float)


# TODO:
# **split camelCase variables
# separate code keywords (private, void, for, int) from non-keywords
def process(s):
	#print("=======")
	#print(s)
	#print()
	s = s.replace('(', ' ').replace(')', ' ')
	s = s.replace('\n', ' ').replace('\t', '')
	# starter code sometimes has a /** 1a **/ with the problem number
	# unclear if this is removing important comments though
	if '/**' in s:
		s = s[s.find('**/') + 3:].strip()
	# remove other punctuation (e.g. {, =, *)
	s = re.sub(r'[^\w\s]', '', s)
	# remove extraneous whitespace
	s = re.sub(' +', ' ', s)

	#print(s)
	#print("=======")
	return s

def CamelCaseProcess(s):
	return s


def main():
	# TODO: better train / val / test split
	indices = list(range(len(data)))

	random.shuffle(indices)
	split = int(0.8 * len(indices))
	trainIndices, testIndices = indices[:split], indices[split:]
	trainFeatures = []
	testFeatures = []
	for i in trainIndices:
		trainFeatures.append(process(answers[i]))
	for i in testIndices:
		testFeatures.append(process(answers[i]))
	print("Transforming answers")
	cv = CountVectorizer()
	trainX = cv.fit_transform(trainFeatures)
	testX = cv.transform(testFeatures)
	print("Train size: {} Test size: {}".format(trainX.shape, testX.shape))
	trainY, testY = scores[trainIndices], scores[testIndices]

	print("Training")
	model = LinearRegression()
	model.fit(trainX, trainY)
	train_r2 = model.score(trainX, trainY)
	test_r2 = model.score(testX, testY)
	print("Train R2: {}".format(train_r2))
	print("Test R2: {}".format(test_r2))


if __name__ == '__main__':
	main()