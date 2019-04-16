import re
import collections


def tokens(text):
    """
    get all words from corpus
    :param text:
    :return:
    """
    return re.findall('[a-z]+', text.lower())

WORDS = tokens(open('big.txt').read())
WORD_COUNTS = collections.Counter(WORDS)


def edits0(word):
    """
    returns all strings that are zero edits away - or the word itself
    :param word:
    :return:
    """
    return{word}


def edits1(word):
    """
    return all strings that are one edit away
    :param word:
    :return:
    """

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def splits(word):
        """
        return a list of all possible (first, rest) pairs
        that the input word is made from
        :param word:
        :return:
        """
        return[(word[:i], word[i:])
               for i in range(len(word)+1)]
    pairs = splits(word)
    deletes = [a + b[1:] for (a, b) in pairs if b]
    transposes = [a + b[1] + b[0] + b[2:] for (a,b) in pairs if len(b) > 1]
    replaces = [a + b + c[1:] for(a,b) in pairs for c in alphabet
                if b]
    inserts = [a + c + b for(a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """
    return all strings that are two edits away from the input word
    :param word:
    :return:
    """
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


def known(words):
    """
    return the subset of words that are actually in WORD_COUNTS dictionary
    :param words:
    :return:
    """
    return {w for w in words if w in WORD_COUNTS}


# inferior method as it does not account for varying case
def correct(word):
    """ get the best spelling for the input word
    :param word:
    :return:
    """
    canidates = (known(edits0(word)) or
                 known(edits1(word)) or
                 known(edits2(word)) or
                 [word])
    return max(canidates, key=WORD_COUNTS.get)


def correct_match(match):
    word = match.group()

    def case_of(text):
        """
        return case function appropriate:
        upper, lower, title, or just str
        :param text:
        :return:
        """
        return(str.upper if text.isupper() else
               str.lower if text.islower() else
               str.title if text.istitle() else
               str)
    return case_of(word(correct(word.lower())))


def correct_text_generic(text):
    """
    correct all the words within a text,
    returning the corrected text
    :param text:
    :return:
    """
    return re.sub('[a-zA-Z]+', correct_match, text)

# regarding spelling also consider PyEnchant and aspell - python

# the same algorithm as seen above (correct) is implemented in pattern as from pattern.en import suggest
# ----> suggest(word)

#
