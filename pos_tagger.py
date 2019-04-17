import nltk
from nltk.corpus import treebank
from nltk.tag.sequential import ClassifierBasedPOSTagger
from nltk.classify import NaiveBayesClassifier
import pickle


"""
a custom parts of speech tagger following design and techniques using in 'Text Analytics in Python'
by: Dipanjan Sarkar
"""

# for testing purposes, tagger will eventually be passed tokens from preprocess
#sentence = "The brown fox id quick and he is jumping over the lazy dog"
#tokens = nltk.word_tokenize(sentence)

################################################################

# get data and training data - GLOBAL
data = treebank.tagged_sents()
train_data = data[:3500]
test_data = data[3500:]


nbt = ClassifierBasedPOSTagger(train=train_data, classifier_builder=NaiveBayesClassifier.train)
save_classifier = open('ClassifierBasedPOSTagger.pickle', 'wb')
pickle.dump(nbt, save_classifier)
save_classifier.close()


