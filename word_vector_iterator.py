import math
import numpy as np
import codecs
from keras.preprocessing.image import Iterator

class ClienWordVectorIterator(Iterator):

	def __init__(self, input_filename, vector_filename, batch_size, shuffle = True, seed = None, train_mode = True):
		data = []
		classes = []
		cnt = 0

		f = codecs.open(input_filename, "r", "utf-8")

		for line in f:
			if line.strip() != "":
				cnt += 1
				tokens = line.strip().split("\t")

				if len(tokens) < 3:
					continue

				if train_mode and cnt <= 90000:
					data.append((tokens[2], tokens[0]))
				elif not train_mode and cnt > 90000:
					data.append((tokens[2], tokens[0]))
				
				if not tokens[0] in classes:
					classes.append(tokens[0])

		f.close()

		vectorMap = {}
		self.vectorDim = 0

		f = codecs.open(vector_filename, "r", "utf-8")

		for line in f:
			if line.strip() != "":
				tokens = line.strip().split("\t")
				w = tokens[0]
				v = map(float, tokens[1:])

				vectorMap[w] = v
				self.vectorDim = len(v)

		f.close()

		self.data = data
		self.vectorMap = vectorMap
		self.classes = sorted(classes)

		N = len(data)

		print "%d samples are loaded." % len(data)

		super(ClienWordVectorIterator, self).__init__(N, batch_size, shuffle, seed)

	def next(self):
		with self.lock:
			index_array, current_index, current_batch_size = next(self.index_generator)

		vectorMap = self.vectorMap

		batch_x = []
		batch_y = []

		for i, j in enumerate(index_array):
			x, y = self.data[j]
			
			xx = []
			yy = [0.] * len(self.classes)

			for i in xrange(100):
				xx.append([0] * self.vectorDim)

			for k, w in enumerate(x.split(" ")):
				if vectorMap.get(w) != None and k < len(xx):
					xx[k] = vectorMap[w]

			yy[self.classes.index(y)] = 1.

			batch_x.append([xx])
			batch_y.append(yy)

		return np.array(batch_x), np.array(batch_y)

