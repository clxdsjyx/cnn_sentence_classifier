from deeplearning_assistant.assistant import DeepLearningAssistant
from word_vector_iterator import ClienWordVectorIterator
#from tmon_word_vector_iterator import TmonSentimentAnalysisWordVectorIterator as TmonIterator
from cnn_classifier import CnnClassifier

if __name__ == "__main__":
	assistant = DeepLearningAssistant("./hparams.json")
	mb = CnnClassifier()
	
	trainIterator = ClienWordVectorIterator("input.txt", "vectors.txt", assistant.BATCH_SIZE, train_mode = True)
	testIterator = ClienWordVectorIterator("input.txt", "vectors.txt", assistant.BATCH_SIZE, train_mode = False)

	#trainIterator = TmonIterator("input.txt", "vectors.txt", assistant.BATCH_SIZE, train_mode = True)
	#testIterator = TmonIterator("test.txt", "vectors.txt", assistant.BATCH_SIZE, train_mode = False)

	assistant.train(mb, trainIterator, testIterator)
