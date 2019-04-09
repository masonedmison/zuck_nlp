import nltk
import re
import string

# for testing purposes
file = '/Users/MasonBaran/Desktop/xml_read_test/2019-006.xml'


def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
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

