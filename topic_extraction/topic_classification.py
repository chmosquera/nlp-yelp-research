import nltk, os
from nltk import Tree
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
from topic_extraction.topic_classification_model import YelpTopicClassification


def loadData(self):
    global FOOD_VOCAB, ATMSOSPHERE_VOCAB, SERVE_VOCAB, PRICE_VOCAB
    path = os.path.join(os.path.dirname(__file__), "..\\data\\foods.txt")
    data = open(path)
    FOOD_VOCAB = data.readlines()[0].split(',')
    FOOD_VOCAB = list(set(self.food_vocab))
    data.close()

    path = os.path.join(os.path.dirname(__file__), "..\\data\\atmosphere.txt")
    data = open(path)
    ATMSOSPHERE_VOCAB = data.readlines()[0].split(',')
    ATMSOSPHERE_VOCAB = list(set(self.atmosphere_vocab))
    data.close()

    path = os.path.join(os.path.dirname(__file__), "..\\data\\service.txt")
    data = open(path)
    SERVE_VOCAB = data.readlines()[0].split(',')
    SERVE_VOCAB = list(set(self.service_vocab))
    data.close()

    path = os.path.join(os.path.dirname(__file__), "..\\data\\price.txt")
    data = open(path)
    PRICE_VOCAB = data.readlines()[0].split(',')
    PRICE_VOCAB = list(set(self.price_vocab))
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

def filterTokens(tokenized):
    filtered = []
    for token in tokenized:
        if token.isalpha() and token.lower() not in stopwords.words('english'):
            filtered.append(token)  # deacc=True removes punctuations
    return filtered

def lemmatize(tagged):
    lemmas = []
    for token,tag in tagged:
        wn_tag = getWordnetPOS(tag)
        if wn_tag:
            lemmas.append(lemmatizer.lemmatize(token, pos=wn_tag))
    return lemmas


def extractPhrases2(review):
    tokenized = nltk.word_tokenize(review.lower())
    tagged = nltk.pos_tag(tokenized)

    grammar = r"""
      NP: {<DT>?<JJ>*<NN|NN|NNP|NNPS>}          # Chunk sequences of DT, JJ, NN
      PP: {<IN><NP>}               # Chunk prepositions followed by NP
      VP: {<VB|VBG|VBD|VBN|VBP|VBZ.*><NP|PP>} # Chunk verbs and their arguments
      # CLAUSE: {<NP><VP>}           # Chunk NP, VP
      """
    chunker = nltk.RegexpParser(grammar)
    tree = chunker.parse(tagged)
    firstTime = True
    # for subtree in tree.subtrees():
    Tree.fromstring(str(tree)).pretty_print()

def regexChunker(sentence):
    grammar = r"""
      NP: {<DT>?<JJ>*<NN|NN|NNP|NNPS>}          # Chunk sequences of DT, JJ, NN
      PP: {<IN><NP>}               # Chunk prepositions followed by NP
      VP: {<VB|VBG|VBD|VBN|VBP|VBZ.*><NP|PP>} # Chunk verbs and their arguments
      # CLAUSE: {<NP><VP>}           # Chunk NP, VP
      """
    chunked = nltk.RegexpParser(grammar)
    return chunked.parse(sentence)

# Takes a list of lemmatized tokens, returns a list of (text, grammar label) pairs of phrases
def extractPhrases(review, debug=False):
    tokenized = nltk.word_tokenize(review)
    tagged = nltk.pos_tag(tokenized)
    tree = regexChunker(tagged)
    phrases = []
    for subtree in tree.subtrees():
        if subtree.label() in ["NP", "PP", "VP", "CLAUSE", "NOUN"]:
            p = " ".join([tag[0] for tag in subtree.leaves()])
            # print(p, subtree.label())
            phrases.append((p,subtree.label()))
    return phrases

def extractTopics(review):
    # preprocess raw text
    tokenized = nltk.word_tokenize(review)
    filtered = filterTokens(tokenized)
    tagged = nltk.pos_tag(filtered)
    print("tagged: ", tagged)

    noun_tags = []
    for tok,tag in tagged:
        if tag.startswith('N'):
            noun_tags.append((tok,tag))

    print("nouns: ", noun_tags)
    lemma_nouns = lemmatize(noun_tags)

    # Also add nouns from the noun phrases (some nouns are only detected in noun phrases)
    phrases = extractPhrases(review)
    for phrase,tag in phrases:
        # underscore = "_".join(phrase.split())
        # Check if any of the words in the phrase are nouns
        lemma_nouns.extend(phrase.split())
    print("phrases: ", phrases)

    lemma_nouns = list(set(lemma_nouns))    # no duplicates
    print("lemma_nouns: ", lemma_nouns)

    # check the nouns against our bow for each topic
    topics = []
    for noun in lemma_nouns:
        noun = noun.lower()
        if noun in FOOD_VOCAB:
            topics.append((noun, "<FOOD>"))
        if noun in ATMOSPHERE_VOCAB:
            topics.append((noun, "<ATMSOSPHERE>"))
        if noun in PRICE_VOCAB:
            topics.append((noun, "<PRICE>"))
        if noun in SERVICE_VOCAB:
            topics.append((noun, "<SERVICE>"))

    return topics


def main():
    loadData()


# Global vars
FOOD_VOCAB = []
ATMOSPHERE_VOCAB = []
SERVICE_VOCAB = []
PRICE_VOCAB = []

if __name__ == '__main__':
    main()

# print(samples[2])
# topics = extractTopics(samples[2])
# print("TOPICS: \n")
# print(topics)

## Test
# samples =
# extractPhrases2(samples[0])
# topics = extractTopics(samples[0])
# print(topics)