import nltk
import re
import string
from contractions import CONTRACTION_MAP

# for testing purposes
#file = '/Users/MasonBaran/Desktop/xml_read_test/2019-006.xml'

# short test corpus
corpus = "The brown fox wasn't that quick and he couldn't win the race \
          Hey that's a great deal! I just bought a phone for $199\
          @@You'll (learn) a **lot** in the book. Python is an amazing language!@@"


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


def expand_contractions(sentence, contraction_mapping):
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
    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence


if __name__ == '__main__':
    # sent tokenize
    sentences = nltk.sent_tokenize(corpus)

    # test remove spec. char before full tokenize
    clean_sents = [remove_characters_before_tokenization(sent, keep_apostrophes=True) for sent in sentences]

    # test expand_contractions
    expanded_sentences = [expand_contractions(sent, CONTRACTION_MAP) for sent in clean_sents]

    print("After: {}".format(expanded_sentences))