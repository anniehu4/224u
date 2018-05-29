import pickle
import re
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pickle.load(open('data.pkl', 'rb'))
answers = [d[0] for d in data]
scores = np.array([d[2] for d in data])

def splitCamelCase(s):
	words = s.split(' ')
	new_s = []
	for i in range(len(words)):
		w = words[i]
		new_w = []
		if not w or w[0].isupper():
			new_s.append(w)
			continue
		start = 0
		for i in range(1, len(w)):
			if w[i].isupper():
				new_w.append(w[start:i].lower())
				start = i
		if start > 0:
			new_w.append(w[start:].lower())
			new_s.extend(new_w)
		else:
			new_s.append(w)
	return ' '.join(new_s)

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


def main():
	# TODO: better train / val / test split
	indices = range(len(data))
	random.seed(1)
	random.shuffle(indices)
	split = int(0.8 * len(indices))
	trainIndices, testIndices = indices[:split], indices[split:]
	trainFeatures = []
	testFeatures = []
	print "Processing strings"
	for i in trainIndices:
		trainFeatures.append(process(answers[i]))
	for i in testIndices:
		testFeatures.append(process(answers[i]))
	print "Transforming answers"
	cv = CountVectorizer()
	trainX = cv.fit_transform(trainFeatures)
	testX = cv.transform(testFeatures)
	print "Train size:", trainX.shape, "Test size:", testX.shape
	trainY, testY = scores[trainIndices], scores[testIndices]

	print "Training"
	model = LinearRegression(fit_intercept=False)
	model.fit(trainX, trainY)
	print "Train R2:", model.score(trainX, trainY)
	print "Test R2:", model.score(testX, testY)

	predictions = abs(model.predict(testX))
	plt.scatter(testY, predictions)
	plt.show()



if __name__ == '__main__':
	main()