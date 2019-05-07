from skmultilearn.problem_transform import LabelPowerset
from sklearn.linear_model import SGDClassifier
from preprocess import normalize_corpus
from xml_parse import xml_parse
from sklearn.model_selection import train_test_split


"""
ON HOLD - implement keyphrase extraction on documents first - these may be helpful in divising labels for 
supervised ML algos
Testing label powerset with various multi-class classifiers
"""

# testing data set in xml_parse

corpus, titles, record_ids, df = xml_parse()
nc = normalize_corpus(corpus)


# split train and test sets where X = features and y = labels -- use sklearn



# initialize label powerset multi-label classifier
classifier = LabelPowerset(SGDClassifier(loss='hinge'), n_iter=1000)