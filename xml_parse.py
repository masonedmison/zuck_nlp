import xml.etree.ElementTree as ET
import glob
from preprocess import normalize_corpus
import os
from utils import build_feature_matrix

'''
pratice run
tasks:
    read in xml file(s)
    ?
'''

CORPUS = []


def set_directory():
    # real training set
    # os.chdir('/Users/MasonBaran/Desktop/zuck_tests_to_to_cluster')
    # TESTING
    os.chdir('/Users/MasonBaran/Desktop/TESTING')


def get_files():
    for file_names in glob.iglob('*.xml'):
        yield file_names


# parse xml at participant level
def parse_xml_IP(file):
    """ parses xml files searching for each participant tag, keywords returned from preprocessing are appended
    as participant attributes
    :argument
    xml file
    """

    if file.split('.')[1] != "xml":
        raise TypeError("{} is not an .xml file".format(file))
    # create tree
    tree = ET.parse(file)
    # get base root of tree instance
    root = tree.getroot()

    for child in root.iter():
        if child.tag == 'participant':
            normalize_corpus(child.text)
            
            # append keywords as attribute using child.set for each attribute
            # child.set("test", ["ing", "ing2"])
            # print(child.attrib)


def parse_xml_ff(file):
    """ ff = full file -- parses xml files and joins all participant utterances into single string
    The resulting string is sent to preprocessing
    :argument xml file
    """
    # TODO replace all items in square brackets with "" - [] indicates non-spoken elements
    if file.split('.')[1] != "xml":
        raise TypeError("{} is not an .xml file".format(file))
    
    tree = ET.parse(file)
    root = tree.getroot()

    contents_whole = [child.text for child in root.iter() if child.tag == 'participant']
    # join with space between participant utterances
    contents_joined = " ".join(contents_whole)
    CORPUS.append(contents_joined)


if __name__ == "__main__":
    set_directory()
    files = get_files()
    # for file in files:
    #     parse_xml_IP(file)

    # parse xml for individual participant
    # parse_xml_IP(file)

    # parse xml for full participant contents
    for file in files:
        parse_xml_ff(file)
        print(file)

    # place holder 'driver' logic

    nc = normalize_corpus(CORPUS)
    vectorizer, feature_matrix = build_feature_matrix(CORPUS,
                                                      feature_type='tfidf',
                                                      min_df=0.24, max_df=0.85,
                                                      ngram_range=(1,2))
    print(vectorizer)
