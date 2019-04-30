import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import re
import string
from contractions import CONTRACTION_MAP
import pickle

# file for testing purposes
#file = '/Users/MasonBaran/Desktop/xml_read_test/2019-006.xml'


# short corpus for testing
corpus = ["The brown fox wasn't that quick and he couldn't win the race \
          Hey that's a great deal! I just bought a phone for $199\
          @@You'll (learn) a **lot** in the book. Python is an amazing language!@@",
            "Hi my names is johnny guamy, I like eggs.... so much it makes me an eggspert, Please. Thank you?",
          "I ran run runned as fast I couldn't. The boy swear swore sworn too"]


def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def remove_characters_before_tokenization(text, keep_apostrophes = False):
    text = text.strip()
    if keep_apostrophes:
        pattern = r'[?|$|&|*|%|@||(|)|~]' # add any characters to remove
        filtered_sentence = re.sub(pattern, r'', text)
    else:
        pattern = r'[^a-zA-Z0-9]' #only extract alpha numeric characters
        filtered_sentence = re.sub(pattern, r' ', text)
    return filtered_sentence


def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens


def expand_contractions(text, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopwords]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wn.synsets(old_word):
            return old_word
        correct_tokens = [replace(word) for word in tokens]
        return correct_tokens


# stemming - porter stemmer implementation
def stem_tokens(tokens):
    ps = nltk.PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens


def get_tagger():
    """
    opens pickle file and returns trained classifier object
    :return: classifier
    """
    classifier_f = open('ClassifierBasedPOSTagger.pickle', 'rb')
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


def pos_tag_text(text):

    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    tagger = get_tagger()
    tagged_text = tagger.tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text


def lemmatize(text):
    wnl = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    pos_tagged_text = pos_tag_text(tokens)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag else word for word, pos_tag in pos_tagged_text]
    lemmatized_text = " ".join(lemmatized_tokens)
    return lemmatized_text


def normalize_corpus(corpus, tokenize=False):
    normalized_corpus = []
    for text in corpus:
        text = expand_contractions(text, CONTRACTION_MAP)
        text = lemmatize(text)
        # all good
        text = remove_characters_before_tokenization(text)
        text = remove_stopwords(text)
        if tokenize:
            text = tokenize_text(text)
        normalized_corpus.append(text)



if __name__ == '__main__':
    normalize_corpus(corpus)







