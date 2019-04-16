import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import re
import string
from contractions import CONTRACTION_MAP
import pickle

# for testing purposes
#file = '/Users/MasonBaran/Desktop/xml_read_test/2019-006.xml'
# load in stopwords from nltk


# short test corpus
corpus = "The brown fox wasn't that quick and he couldn't win the race \
          Hey that's a great deal! I just bought a phone for $199\
          @@You'll (learn) a **lot** in the book. Python is an amazing language!@@"


def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence).strip() for sentence in sentences]
    return word_tokens


def remove_characters_before_tokenization(sentence, keep_apostrophes = False):
    sentence = sentence.strip()
    if keep_apostrophes:
        pattern = r'[?|$|&|*|%|@||(|)|~]' # add any characters to remove
        filtered_sentence = re.sub(pattern, r'',sentence)
    else:
        pattern = r'[^a-zA-Z0-9]' #only extract alpha numeric characters
        filtered_sentence = re.sub(pattern, r'', sentence)
    return filtered_sentence


def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None,[pattern.sub('', token) for token in tokens])
    return filtered_tokens


def expand_contractions(text, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.DOTALL|re.IGNORECASE)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
            if contraction_mapping.get(match)\
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'","", expanded_text)
    return expanded_text


def remove_stopwords(tokens):
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens


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


def tag_tokens(tokens):
    tagger = get_tagger()
    return tagger.tag(tokens)


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

def lemmatize(text):
    wnl = WordNetLemmatizer()
    pos_tagged_text = tag_tokens(text)
    lemmatized_tokens = [wnl.lemmatize(word,pos_tag) if pos_tag else word for word, pos_tag in pos_tagged_text]
    lemmatized_text = " ".join(lemmatized_tokens)
    return lemmatized_text


def normalize_corpus(corpus, tokenize=False):
    normalized_corpus = []
    for text in corpus:
        text = expand_contractions(text, CONTRACTION_MAP)
        text = lemmatize(text)
        normalized_corpus.append(text)
    #for testing
    print(normalized_corpus)


if __name__ == '__main__':
    normalize_corpus(corpus)







