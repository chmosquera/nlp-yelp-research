import nltk, spacy
from nltk.corpus import sentiwordnet
from nltk.corpus import wordnet
from nltk.wsd import lesk
from nltk.tokenize.casual import casual_tokenize
from nltk import word_tokenize

nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('stopwords')

spacy_lg_path = 'D:\\Programs\\Python37x64\\nlp_config\\venv\\Lib\\site-packages\\en_core_web_lg\\en_core_web_lg-2.2.5'
nlp = spacy.load(spacy_lg_path)


# casual tokenize works better with sentiwordnet for what I need
# e.g. 'the food was good'. With regular tokenizing, "was"'s sense is "washington"....weird?
# with casual tokenizing, "was"'s sense is "be" which is correct
def casualTokenize(raw_sentence, preserve_case=False):
  return casual_tokenize(raw_sentence, preserve_case=preserve_case, reduce_len=True)

def getWordnetPOS(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def loadSentimentLexicons():
    global BING_NEG_WORDS, BING_POS_WORDS
    f = open('D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\\data\\opinion-lexicon-English\\negative-words.txt', 'r')
    lines = f.readlines()
    BING_NEG_WORDS = [line.replace('\n', "") for line in lines if not line[0] == ';']
    f.close()

    f = open('D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\\data\\opinion-lexicon-English\\positive-words.txt', 'r')
    lines = f.readlines()
    BING_POS_WORDS = [line.replace('\n', "") for line in lines if not line[0] == ';']
    f.close()


def wsdLesk(sentence, word, wn_pos=None, debug=False):
    tokenized = casualTokenize(sentence)
    wsd_syn = None

    # lesk without pos
    if not wn_pos:
        wsd_syn = lesk(tokenized, word)
    # lesk with pos
    else:
        wsd_syn = lesk(tokenized, word, pos=wn_pos)

        # if the pos is incorrect, try lesk without the pos
        if wsd_syn is None:
            wsd_syn = lesk(tokenized, word)
    if debug and wsd_syn is not None:
      print(word, wsd_syn, wsd_syn.definition())
    return wsd_syn


def calculate_sentiment(syn_name, use_bing=True):
    global BING_NEG_WORDS, BING_POS_WORDS

    senti_syn = sentiwordnet.senti_synset(syn_name)
    pos,neg,obj,name = senti_syn.pos_score(), senti_syn.neg_score(), senti_syn.obj_score(), senti_syn.synset.lemmas()[0].name()
    if use_bing:
        if name in BING_NEG_WORDS:              # Use Hu and Bing Liu's lexicon to determine orientation (if applicable)
            if neg == 0:                        # Further, if the sentiWordNet score is 0, but it has orientation,
                neg = (1.1-obj)               # let it's pos/neg score be dependent on word's objectivity.
            return round(-neg,3)
        elif name in BING_POS_WORDS:
            if pos == 0:
                pos = (1.1-obj)
            return round(pos,3)

    if obj > 0.8:             # objectivity threshold
        return 0.0
    return round(pos-neg,3)


def sentimentAnalysis(sentence, included_pos=[wordnet.ADJ, wordnet.ADV, wordnet.VERB], use_bing=True, soft=True):
    sentence = sentence.lower()

    # convert to spacy doc
    doc = nlp(sentence)

    # sum the total sentiment score
    total_sent = 0
    total_toks = 0
    for tok in doc:
        word = tok.text

        # convert to wordnet pos
        wntag = getWordnetPOS(tok.tag_)

        if wntag not in included_pos:
            continue

        # use the common sense of each word
        syns = wordnet.synsets(word,pos=wntag)
        if len(syns) == 0:
            continue
        syn = syns[0]

        # calculate the sentiment of this sense, add to total
        score = calculate_sentiment(syn.name(), use_bing=use_bing)
        total_sent += score
        total_toks += 1
        # print(word, score)

    # calculate average sentiment
    if total_toks == 0:
        avg_sent = 0
    else:
        avg_sent = total_sent/total_toks

    if not soft:
        return normalizeSentiments(avg_sent)

    return avg_sent


# With lesk
def sentimentAnalysisLesk(sentence, included_pos=[wordnet.ADJ, wordnet.ADV, wordnet.VERB], use_bing=True, soft=True):
    sentence = sentence.lower()

    # convert to spacy doc
    doc = nlp(sentence)

    # sum the total sentiment score
    total_sent = 0
    total_toks = 0
    for tok in doc:
        word = tok.text

        # convert to wordnet pos
        wntag = getWordnetPOS(tok.tag_)

        if wntag not in included_pos:
            continue

        # use the common sense of each word
        # lesk_syn = wsdLesk(sentence, word, wntag)
        lesk_syn = lesk(sentence, word, wntag)
        if lesk_syn:        	

            # calculate the sentiment of this sense, add to total
            score = calculate_sentiment(lesk_syn.name(), use_bing=use_bing)
            total_sent += score
            # print(word, score)
            total_toks += 1         # total tokens with sentiment

    # calculate average sentiment
    if total_toks == 0:
        avg_sent = 0
    else:
        avg_sent = total_sent/total_toks

    if not soft:
        return normalizeSentiments(avg_sent)

    return avg_sent


def normalizeSentiments(score):
    if score == 0:
        return 0
    elif score > 0:
        return 1
    else:
        return -1


def main():
    loadSentimentLexicons()


BING_NEG_WORDS = []
BING_POS_WORDS = []

if __name__ == "__main__":
    main()


