from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
import numpy as np
import csv


def build_feature_matrix(documents, feature_type='frequency', min_df=0.0, max_df=1.0, ngram_range = (1,1)):
    feature_type = feature_type.lower().strip()

    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=min_df, max_df=max_df,
                                     ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df, max_df=max_df,
                                     ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                     ngram_range=ngram_range)
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)

    return vectorizer, feature_matrix


def get_metrics(true_labels, predicted_labels):
    print(
        'Accuracy:', np.round(
            metrics.accuracy_score(true_labels,
                                   predicted_labels),
            2))
    print(
        'Precision:', np.round(
            metrics.precision_score(true_labels,
                                    predicted_labels,
                                    average='weighted'),
            2))
    print(
        'Recall:', np.round(
            metrics.recall_score(true_labels,
                                 predicted_labels,
                                 average='weighted'),
            2))
    print(
        'F1 Score:', np.round(
            metrics.f1_score(true_labels,
                             predicted_labels,
                             average='weighted'),
            2))


def write_to_csv(dict, file_name, fieldnames=None):
    with open('/Users/MasonBaran/zuck_nlp/files/csv/{}.csv'.format(file_name), 'w') as outfile:
        w = csv.DictWriter(outfile, fieldnames=fieldnames)
        if fieldnames:
            w.writeheader()
        # if nested_dicts:
        #     for d in dict:
        #         print(type(dict[d]))
        #         w.writerows(dict(dict[d]))
        # else:
        #     w.writerows(dict)
        w.writerows(dict)
