from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
import gensim
import nltk


CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]

new_doc = ['loving this blue sky today']


def tfidf_extractor(corpus, ngram_range=(1,1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def display_features(features, feature_names):
    df = pd.DataFrame(data=features,
                      columns=feature_names)
    print(df)

################################################################
# experimentation with gensim

# temp tokenization


tokenized_corpus = [nltk.word_tokenize(sent) for sent in CORPUS]
tokenized_new_doc = nltk.word_tokenize(new_doc)

model = gensim.models.Word2Vec(tokenized_corpus, size=10, window=10, min_count=2, sample=1e-3)








