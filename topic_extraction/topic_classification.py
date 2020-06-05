import nltk, os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
import spacy
nlp = spacy.load("en_core_web_sm")


def loadData():
    global FOOD_VOCAB, ATMOSPHERE_VOCAB, SERVICE_VOCAB, PRICE_VOCAB
    path = os.path.join(os.path.dirname(__file__), "..\\data\\foods.txt")
    print(path)
    data = open(path)
    FOOD_VOCAB = data.readlines()[0].split(',')
    FOOD_VOCAB = list(set(FOOD_VOCAB))
    data.close()

    path = os.path.join(os.path.dirname(__file__), "..\\data\\atmosphere.txt")
    data = open(path)
    ATMSOSPHERE_VOCAB = data.readlines()[0].split(',')
    ATMSOSPHERE_VOCAB = list(set(ATMSOSPHERE_VOCAB))
    data.close()

    path = os.path.join(os.path.dirname(__file__), "..\\data\\service.txt")
    data = open(path)
    SERVICE_VOCAB = data.readlines()[0].split(',')
    SERVICE_VOCAB = list(set(SERVICE_VOCAB))
    data.close()

    path = os.path.join(os.path.dirname(__file__), "..\\data\\price.txt")
    data = open(path)
    PRICE_VOCAB = data.readlines()[0].split(',')
    PRICE_VOCAB = list(set(PRICE_VOCAB))
    data.close()


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

def lemmatize(tagged):
    lemmas = []
    for token,tag in tagged:
        wn_tag = getWordnetPOS(tag)
        if wn_tag:
            lemmas.append(lemmatizer.lemmatize(token, pos=wn_tag))
    return lemmas


def aspectOpinionPair(sentence):
    doc = nlp(sentence)
    pairs = []

    # don't include words we've already seen. (Some subjects share predicates)
    history = []

    # every pair needs a subject
    for chunk in doc.noun_chunks:
        noun = chunk.root
        relevant_words = []

        # Find modifiers describing the noun
        MODS = ['amod', 'advmod']
        relevant_words.extend([child for child in chunk.root.children if child.dep_ in MODS])

        # Find any predicates of this subject (verbs)
        predicate = ""
        pred_children = []
        if str(chunk.root.head.tag_).startswith('V') and chunk.root.head not in history:
            predicate = chunk.root.head
            relevant_words.append(predicate)
            relevant_words.extend([child for child in predicate.children if not child == chunk.root if child.text])
            history.append(predicate)

        pairs.append((noun.text, relevant_words, assignTopic(noun.text)))
    return pairs

def assignTopic(word):
    global FOOD_VOCAB, ATMOSPHERE_VOCAB, SERVICE_VOCAB, PRICE_VOCAB
    tagged = nltk.pos_tag([word])
    wntag = getWordnetPOS(tagged[0][1])
    if wntag is None:
        return None
    lemma = lemmatizer.lemmatize(word, pos=wntag)
    # print(word, tagged, lemma, wntag)

    syns = wordnet.synsets(lemma)

    # build a list of synonyms to match the word to
    synonyms = []
    for syn in syns:
        possible_syns = []
        possible_syns.extend([l.name() for l in syn.lemmas()])
        possible_syns.extend([h.lemmas()[0].name() for h in syn.hyponyms()])
        possible_syns.extend([h.lemmas()[0].name() for h in syn.hypernyms()])

        synonyms.append((syn,possible_syns))
    # print(synonyms)

    # the word might appear in a few categories, get the one that has the highest score
    score = {
        'food':0,
        'atmosphere':0,
        'service':0,
        'price':0,
        'other':0
    }
    for syn,possible_syns in synonyms:
        for s in possible_syns:         # search until one of the words can be categorized
            s = s.lower()
            if s in FOOD_VOCAB:
                score['food'] += 1
                continue
            if s in ATMOSPHERE_VOCAB:
                score['atmosphere'] += 1
                continue
            if s in PRICE_VOCAB:
                score['price'] += 1
                continue
            if s in SERVICE_VOCAB:
                score['service'] += 1
                continue

    max_topic = 'other'
    for key,val in score.items():
        if val > score[max_topic]:
            max_topic = str(key)

    return max_topic

def main():
    loadData()
    print(assignTopic('place'))


# Global vars
FOOD_VOCAB = []
ATMOSPHERE_VOCAB = []
SERVICE_VOCAB = []
PRICE_VOCAB = []

if __name__ == '__main__':
    main()