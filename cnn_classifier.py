from deeplearning_assistant.model_builder import AbstractModelBuilder

class CnnClassifier(AbstractModelBuilder):

	def buildModel(self):
		from keras.models import Model
		from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout

		dropRate = 0.5
		filterSize = [2, 5, 10, 20]
		numberOfFilters = 200
		poolSize = [50, 20, 10, 5]
		denses = []

		input_layer = Input(shape = (1, 100, 200))
		
		for f, p in zip(filterSize, poolSize):
			cnn = Convolution2D(numberOfFilters, f, 200, border_mode = 'valid', activation = 'relu')(input_layer)
			pooling = MaxPooling2D(pool_size = (p, 1))(cnn)
			dropout = Dropout(dropRate)(pooling)
			flatten = Flatten()(dropout)
			dense = Dense(256, activation = 'relu')(flatten)

			denses.append(dense)

		for i in xrange(10):
			cnn = Convolution2D(numberOfFilters, 2, 200, border_mode = 'valid', activation = 'relu', subsample = (1 + i, 1))(input_layer)
			pooling = MaxPooling2D(pool_size = (int(50 / (i + 1)), 1))(cnn)
			dropout = Dropout(dropRate)(pooling)
			flatten = Flatten()(dropout)
			dense = Dense(256, activation = 'relu')(flatten)

			denses.append(dense)

		concat = merge(denses, mode = 'concat', concat_axis = 1)
		dense = Dense(1024, activation = 'relu')(concat)
		dropout = Dropout(dropRate)(dense)
		output_layer = Dense(10, activation = 'softmax')(dropout)
		
		model = Model(input = [input_layer], output = output_layer)
		
		return model
