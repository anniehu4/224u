import pickle
import os
import random


#data = pickle.load(open('data.pkl', 'rb'), encoding='latin1')
direc = 'processed/'
path_dataset = 'processed/data.pkl'
msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
assert os.path.isfile(path_dataset), msg
print("Loading dataset into memory...")
data = pickle.load(open(path_dataset, 'rb'))
print("Parsed {} student answers.".format(len(data)))
print("- done.")

holdout = True

def holdout_split(dev_prob, test_prob):
	dev_holdout = dev_prob + '-cs106aFinalWin18'
	test_holdout = test_prob + '-cs106aFinalWin18'
	train_dataset = []
	dev_dataset = []
	test_dataset = []
	for x in data:
		if dev_holdout == x['question']:
			dev_dataset.append(x)
		elif test_holdout == x['question']:
			test_dataset.append(x)
		else:
			train_dataset.append(x)
	print("Train size: {} Dev size: {} Test size: {}".format(len(train_dataset), len(dev_dataset), len(test_dataset)))
	return train_dataset, dev_dataset, test_dataset
	pickle.dump(train_dataset, open(direc + 'holdout-train.pkl', 'wb'))
	pickle.dump(dev_dataset, open(direc + 'holdout-dev.pkl', 'wb'))
	pickle.dump(test_dataset, open(direc + 'holdout-test.pkl', 'wb'))
	print("- done.")


def main():
	if holdout:
		holdout_split('6', '6')
		return

	random.seed(1)
	random.shuffle(data)

	train_split = int(0.8 * len(data))
	dev_split = int(0.9 * len(data))
	train_dataset, dev_dataset, test_dataset = data[:train_split], data[train_split:dev_split], data[dev_split:]

	print("Train size: {} Dev size: {} Test size: {}".format(len(train_dataset), len(dev_dataset), len(test_dataset)))
	pickle.dump(train_dataset, open(direc + 'train.pkl', 'wb'))
	pickle.dump(dev_dataset, open(direc + 'dev.pkl', 'wb'))
	pickle.dump(test_dataset, open(direc + 'test.pkl', 'wb'))
	print("- done.")



if __name__ == '__main__':
	main()
