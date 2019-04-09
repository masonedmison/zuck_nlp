import xml.etree.ElementTree as ET
import glob
from bs4 import BeautifulSoup
from preprocess import preprocess
import os

'''
pratice run
tasks:
    read in xml file(s)
    ?
'''


def set_directory():
    os.chdir('/Users/MasonBaran/Desktop/xml_read_test')


def getfiles():
    for filenames in glob.iglob('*.xml'):
        yield filenames


# parse xml at participant level
def parse_xml_IP(file):
    """ parses xml files searching for each participant tag, keywords returned from preprocessing are appended
    as participant attributes
    keyword argument:
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
            preprocess(child.text)
            
            # append keywords as attribute using child.set for each attribute
            # child.set("test", ["ing", "ing2"])
            # print(child.attrib)


def parse_xml_FF(file):
    """ parses xml files and joins all participant utterances into single string
    The resultinh string is sent to preprocessing
    :argument xml file
    """
    if file.split('.')[1] != "xml":
        raise TypeError("{} is not an .xml file".format(file))
    
    tree = ET.parse(file)
    root = tree.getroot()

    contents_whole = [child.text for child in root.iter() if child.tag == 'participant']
    # join with space between participant utterances
    contents_join = " ".join(contents_whole)
    preprocess(contents_join)


if __name__ == "__main__":
    set_directory()
    # files = getfiles()
    # for file in files:
    #     parse_xml_IP(file)



    # Testing on single file to send to preprocess
    file = '2019-006.xml'

    # parse xml for individual participant
    parse_xml_IP(file)

    # parse xml for full participant contents
    parse_xml_FF(file)
  