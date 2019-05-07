import xml.etree.ElementTree as ET
import glob
import os
import pandas as pd


CORPUS = []
TITLES = []
R_IDS = []


def set_directory(path):
    # real training set
    os.chdir(path)
    # TESTING
    # os.chdir('/Users/MasonBaran/Desktop/dup_test_train')


def get_files():
    for file_names in glob.iglob('*.xml'):
        yield file_names


"""TODO brainstorm method to capture participant level utterance and also criteria for when utterance is
is 'worth' of this processing, i.e. utterance must be greater than <some length>.
"""

# parse xml at participant level
# def parse_xml_IP(file):
#     """ parses xml files searching for each participant tag, keywords returned from preprocessing are appended
#     as participant attributes
#     :argument
#     xml file
#     """
#
#     if file.split('.')[1] != "xml":
#         raise TypeError("{} is not an .xml file".format(file))
#     # create tree
#     tree = ET.parse(file)
#     # get base root of tree instance
#     root = tree.getroot()
#
#     for child in root.iter():
#         if child.tag == 'participant':
#             normalize_corpus(child.text)
            

def parse_xml_ff(file):
    """ ff = full file -- parses xml files and joins all participant utterances into single string
    The resulting string is sent to preprocessing
    :argument xml file
    """
    # TODO replace all items in square brackets with "" - [] indicates non-spoken elements
    if file.split('.')[1] != "xml":
        raise TypeError("{} is not an .xml file".format(file))
    
    try:
        tree = ET.parse(file)
        root = tree.getroot()

        # get record title
        doc_title_find = root.findall("./metadata/title")
        doc_title = doc_title_find[0].text

        # get record id
        r_id_find = root.findall("./metadata/record_id")
        r_id = r_id_find[0].text

        # add corpus to df

        contents_whole = [child.text for child in root.iter() if child.tag == 'participant']

        # join with space between participant utterances
        contents_joined = " ".join(contents_whole)
        CORPUS.append(contents_joined)
        TITLES.append(doc_title)
        R_IDS.append(r_id)

    except Exception as e:
        print(file)
        print(e.__repr__())


def make_df():
    if len(TITLES) != len(CORPUS) or len(TITLES) != len(R_IDS):
        raise ValueError("title, r_ids, and corpus arrays are not of equal length")
    # else...
    ids = pd.Series(R_IDS)
    titles = pd.Series(TITLES)
    corpus = pd.Series(CORPUS)
    df = pd.DataFrame({'record_id':ids, 'Title': titles, 'corpus': corpus})
    return df


def xml_parse(path):
    set_directory(path)
    files = get_files()
    for file in files:
        parse_xml_ff(file)
    df = make_df()
    return CORPUS, df



