# Convolutional Neural Network Sentence Classification (using Keras)
This repository consists of python codes for sentence classification inspired by this [paper](http://www.aclweb.org/anthology/D14-1181), which describes how to build a sentence classifier using Convolutional Neural Network. Also, this [article](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow) well describes about detail of how it works.

## Dependency
Since this code is implemented based on [Keras](http://keras.io/), you need install Keras before running. Also, it uses my code from [aother repository](https://github.com/kh-kim/deeplearning_assistant) to replace [typical part of deep learning training implementation][1]. So, you can figure out that there are ***import*** for deeplearning_assistant repository codes. Thus, you also need to pull the [repository](https://github.com/kh-kim/deeplearning_assistant) before using it.

## What is differ from the paper
- It needs pre-trained word embedding vector to represent a sentence as an input for neural network.
So, you need to get it before you run. I recommend to use [Gensim Word2vec](https://radimrehurek.com/gensim/models/word2vec.html) library to get vectors, which is one of easiest and trustful library for Word2vec.
- Word vector has 200 dimensions. Of course, you can add channels or replace Word2vec vectors using other ways, such as GloVe.
- The size and number of convolutional filter is differ from the paper, and you can change it easily. You will find out that you need to vary it depends on the Corpus to classify.
- Stride (subsampling) is applied for filters for 2 words. So, we can extract a tuple of words which has long distances in a sentence.
  - For example, if we have a sentence "I ***like*** red and fresh ***apple***.", we can extract (like, apple) tuple using stride with skipping "red and fresh". It might be possible to extract similar pattern using longer filter, but it must be more difficult than this way.

## How to run & evaluation result
After solve the dependency, which is described above, you need to modify *word_vector_iterator.py* to read a data properly, unless you use example data, which is included. The example data is crawled from [Clien](http://clien.net/), and classification task is predicting a proper board topic using a title of post. I've got ***78%*** accuracy for predicting best 1 class for 10 classes (board topics). Also, I've got over ***90%*** for sentiment analysis (predict 1 out of 3) of [TMON](http://www.ticketmonster.co.kr/)'s product reviews (which is not included in this repos).   

### Any comments and suggestion will be appreciated!!

[1]: https://github.com/kh-kim/deeplearning_assistant/blob/master/README.md "How to use deeplearning_assistant"
