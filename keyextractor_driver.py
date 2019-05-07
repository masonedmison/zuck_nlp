from text_rank_4_keyword import TextRank4Keyword
from xml_parse import xml_parse
from preprocess import normalize_corpus

"""
merely to leverage methods and logic in TextRank4Keyword
"""

if __name__ == '__main__':
    corpus, data = xml_parse("/Volumes/My Passport/Zuck Backups/2018-10-24_Zuck_Transcripts/XML")

    # normalize corpus
    normalized_corpus = normalize_corpus(corpus, tokenize=False)

    # append normalize corpus (array of document contents) to data['corpus]
    data['corpus'] = normalized_corpus

    # create instance of TextRank4Keyword
    tr4kw = TextRank4Keyword()

    # where each d is a str text value for each record in dataframe
    all_keywords = []
    for i, d in enumerate(data['corpus']):
        tr4kw.analyze(d, candidate_pos=['NOUN', 'PROPN'], window_size=4, lower=False)
        keys = tr4kw.get_keywords(3, print_keys_only=True)
        all_keywords.append(keys)


    data['keywords'] = all_keywords
    # write data (pd.DataFrame) to excel
    data.to_excel('/Users/MasonBaran/zuck_nlp/files/xls/WHOLE_zuck_keyphrase_normalized.xlsx')
