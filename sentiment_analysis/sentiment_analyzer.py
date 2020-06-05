import nltk, spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet
from nltk import sent_tokenize, word_tokenize, pos_tag

nltk.download('sentiwordnet')
nlp = spacy.load("en_core_web_sm")
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


def sentiWordNet(text, included_pos=['J', 'R', 'V', 'N'], debug=True):
    included_pos = [getWordNetTag(tag) for tag in included_pos]  # convert to wordnet tags
    included_pos = [tag for tag in included_pos if tag]  # remove None
    raw_sentences = sent_tokenize(text.lower())
    sentiment = 0
    tokens_count = 0

    printDebug(debug, "num of sentences: " + str(len(raw_sentences)))
    printDebug(debug, "---")

    for raw_sentence in raw_sentences:
        # tokenized = [tok for tok in word_tokenize(raw_sentence) if tok.isalnum()]
        # tagged_sentence = pos_tag(tokenized)

        doc = nlp(raw_sentence)

        # printDebug(debug, tagged_sentence)
        printDebug(debug, raw_sentence)
        tagged_sentence = [(tok.text, tok.tag_) for tok in doc]
        printDebug(debug, tagged_sentence)

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

            total_sent = 0
            for syn in synsets:
                senti_synset = sentiwordnet.senti_synset(syn.name())
                total_sent += senti_synset.pos_score() - senti_synset.neg_score()
                printDebug(debug, "syn=" + syn.name() + " : senti(pos)=" + str(
                    senti_synset.pos_score()) + " senti(neg): " + str(senti_synset.neg_score()))
            avg_sent = total_sent / len(synsets)

            printDebug(debug, "word=" + word, ", ")
            printDebug(debug, "wntag=" + wntag, ", ")
            printDebug(debug, "lemma=" + lemma, ", ")
            printDebug(debug, "synsets=" + synsets[0].name(), ", ")
            printDebug(debug, "total_senti=" + str(avg_sent))

            if avg_sent != 0:
                sentiment += avg_sent
                tokens_count += 1

        printDebug(debug, "")

    printDebug(debug, "---")
    printDebug(debug, "num_tokens=" + str(tokens_count))
    printDebug(debug, "total_senti=" + str(sentiment))
    return sentiment
