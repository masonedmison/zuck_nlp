import nltk

# for testing purposes
file = '/Users/MasonBaran/Desktop/xml_read_test/2019-006.xml'

def preprocess(text):

    word_tokens = nltk.word_tokenize(text)

    print(word_tokens)