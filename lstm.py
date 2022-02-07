import numpy as np
import heapq
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.layers import LSTM
import pickle
from tensorflow.keras.optimizers import RMSprop
import string

class LSTMLyricGen():
	def __init__(self):
		print("Model Initialized")
		self.songLines = []

	def loadData(self):
		path = './data/baseline/genre_country.txt'
		songLines = []
		textLines = open(path).readlines()
		for index in range(0, len(textLines)):
			if "LYRICS:" in textLines[index]:
				l = 1
				while not("LYRICS:" in textLines[index+l] or "/END LYRICS" in textLines[index+l]):
					# make all lower case and remove whitespace and newlines
					sanitizedLine = textLines[index+l].lower().strip()
					# remove punctuation
					sanitizedLine = sanitizedLine.strip(string.punctuation)
					l += 1
		
		# prints song lines for debugging
		print(" ".join(self.songLines))
		#for line in songLines:
			#print(line)
	
	def processData(self):
		tokenizer = RegexpTokenizer(r'w+')
		wordTokens = tokenizer.tokenize(" ".join(self.songLines))
		uniqueWords = np.unique(wordTokens)
		uniqueWordsIndex = dict((c, i) for i, c in enumerate(uniqueWords))
		print(uniqueWordsIndex)

		

if __name__ == "__main__":
	lGen = LSTMLyricGen()
	lGen.loadData()
	lGen.processData()
