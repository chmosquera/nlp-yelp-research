import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet
from nltk import sent_tokenize, word_tokenize, pos_tag

nltk.download('sentiwordnet')

lemmatizer = WordNetLemmatizer()

def printDebug(debug, msg, end_msg=None):
    if debug:
        print(msg, end=end_msg)

def getWordNetTag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('V'):
        return wordnet.VERB
    return None

def sentiWordNet(text, included_pos=['J', 'R', 'V'], debug=True):
    included_pos = [getWordNetTag(tag) for tag in included_pos] # convert to wordnet tags
    included_pos = [tag for tag in included_pos if tag]         # remove None
    raw_sentences = sent_tokenize(text)
    sentiment = 0
    tokens_count = 0

    printDebug(debug, "num of sentences: " + str(len(raw_sentences)))
    printDebug(debug, "---")

    for raw_sentence in raw_sentences:
        tokenized = [tok for tok in word_tokenize(raw_sentence) if tok.isalnum()]
        tagged_sentence = pos_tag(tokenized)

        printDebug(debug, raw_sentence)

        for word, tag in tagged_sentence:
            wntag = getWordNetTag(tag)

            if wntag not in included_pos:
                continue

            lemma = lemmatizer.lemmatize(word, pos=wntag)
            if not lemma:
                continue

            synsets = wordnet.synsets(lemma, pos=wntag)
            if not synsets:
                continue

            synset = synsets[0]
            senti_synset = sentiwordnet.senti_synset(synset.name())
            word_sent = senti_synset.pos_score() - senti_synset.neg_score()

            printDebug(debug, "word=" + word, ", ")
            printDebug(debug, "wntag=" + wntag, ", ")
            printDebug(debug, "lemma=" + lemma, ", ")
            printDebug(debug, "synsets=" + synsets[0].name(), ", ")
            printDebug(debug, "senti(pos)=" + str(senti_synset.pos_score()) + " senti(neg): " + str(senti_synset.neg_score()), ", ")
            printDebug(debug, "total_senti=" + str(word_sent))

            if word_sent != 0:
                sentiment += word_sent
                tokens_count += 1

        printDebug(debug, "")

    printDebug(debug, "---")
    printDebug(debug, "num_tokens=" + str(tokens_count))
    printDebug(debug, "total_senti=" + str(sentiment))
    return sentiment

import glob, os

samples = []

# os.chdir("./samples")
# for file in glob.glob("*.txt"):
#     f = open(file, 'r')
#     text = f.readlines()
#     samples.append("".join(text))
#
# print(samples[0])
# sentiWordNet(samples[0])

# def sentimentHuLiu():
#     path = os.path.join(os.path.dirname(__file__), "..\\data\\opinion-lexicon-English\\negative-words.txt")
#     f = open(path)
#     neg_words = f.readlines()
#     f.close()
#     neg_words = [word for word in neg_words if not word[0] == ';']
#
#     for word in neg_words:
#         score = sentiWordNet(word, included_pos=['J', 'R', 'V', 'N'])
#         print(word, ": ", score)
#
# sentimentHuLiu()