import pickle
import os
from utils import *

data = pickle.load(open('data/data.pkl', 'rb'))
new_data = []

print("Preparing glove data...")
glove_home = os.path.join('vsmdata', 'glove.6B')
glove_lookup = glove2dict(os.path.join(glove_home, 'glove.6B.50d.txt'))
print(" - done.")

for item in data:
	answer = process(item['answer'], glove_lookup, use_spellcheck=True)
	new_data.append({
			'question': item['question'],
			'answer': answer,
			'score': item['score'], 
			'scoreNormalized': item['scoreNormalized'],
			'rubric': item['rubric'],
			'comment': item['comment']
		})

print(len(data), len(new_data))
pickle.dump(new_data, open('processed/data.pkl', 'wb'))