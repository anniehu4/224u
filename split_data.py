import pickle
import os
import random


#data = pickle.load(open('data.pkl', 'rb'), encoding='latin1')
path_dataset = 'data/data.pkl'
msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
assert os.path.isfile(path_dataset), msg
print("Loading dataset into memory...")
data = pickle.load(open(path_dataset, 'rb'))
print("Parsed {} student answers.".format(len(data)))
print("- done.")

holdout = True
dev_holdout = '6-'
test_holdout = '6-'

def main():
	if holdout:
		train_dataset = []
		dev_dataset = []
		test_dataset = []
		for x in data:
			if dev_holdout in x['question']:
				dev_dataset.append(x)
			elif test_holdout in x['question']:
				test_dataset.append(x)
			else:
				train_dataset.append(x)
		print("Train size: {} Dev size: {} Test size: {}".format(len(train_dataset), len(dev_dataset), len(test_dataset)))
		pickle.dump(train_dataset, open('data/holdout-train.pkl', 'wb'))
		pickle.dump(dev_dataset, open('data/holdout-dev.pkl', 'wb'))
		pickle.dump(test_dataset, open('data/holdout-test.pkl', 'wb'))
		print("- done.")
		return

	random.seed(1)
	random.shuffle(data)

	train_split = int(0.8 * len(data))
	dev_split = int(0.9 * len(data))
	train_dataset, dev_dataset, test_dataset = data[:train_split], data[train_split:dev_split], data[dev_split:]

	print("Train size: {} Dev size: {} Test size: {}".format(len(train_dataset), len(dev_dataset), len(test_dataset)))
	pickle.dump(train_dataset, open('data/train.pkl', 'wb'))
	pickle.dump(dev_dataset, open('data/dev.pkl', 'wb'))
	pickle.dump(test_dataset, open('data/test.pkl', 'wb'))
	print("- done.")



if __name__ == '__main__':
	main()
