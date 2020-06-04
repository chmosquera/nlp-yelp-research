
samples = []

samples.append("""Small unassuming place that changes their menu every so often. Cool decor and vibe inside their 30 seat restaurant. Call for a reservation.

We had their beef tartar and pork belly to start and a salmon dish and lamb meal for mains. Everything was incredible! I could go on at length about how all the listed ingredients really make their dishes amazing but honestly you just need to go.

A bit outside of downtown montreal but take the metro out and it's less than a 10 minute walk from the station.""")

samples.append("""Server was a little rude.

Ordered the calamari, duck confit poutine and the trout fish with miso soba - all very tasty. Definitely not your typical diner.
""")

samples.append("""Hidden on the east end of the Danforth is a lovely Thai restaurant. Found this restaurant while surfing on Yelp.

Surprisingly this place is pretty big to seat a big group of 12 and still have a few more small tables available. They are mainly a take-out restaurant, as many of the customers came in for take-out, and the service was a little bit slow. However, the food was great and it accommodating for many different people - seafood lovers, vegans, etc.

Definitely recommend the basil fried rice and the fried sole fish with eggplant but for those curry lovers out there, stay away from the roti curry chicken - no curry taste, just coconut favour. But the Roti itself was really good.

Most of the dishes were under $10 - definitely a good deal!

""")

import nltk
from nltk import Tree
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
from topic_extraction.topic_classification_model import YelpTopicClassification

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
        if noun in YTC.food_vocab:
            topics.append((noun, "<FOOD>"))
        if noun in YTC.atmosphere_vocab:
            topics.append((noun, "<ATMS>"))
        if noun in YTC.price_vocab:
            topics.append((noun, "<PRICE>"))
        if noun in YTC.service_vocab:
            topics.append((noun, "<SERVICE>"))

    return topics

def extractOpinionWords():
    # Get all the adjectives in the text,
    # https://datascience.stackexchange.com/questions/41285/any-efficient-way-to-find-surrounding-adjective-verbs-with-respect-to-the-target
    return

YTC = YelpTopicClassification()
YTC.loadData()

# print(samples[2])
# topics = extractTopics(samples[2])
# print("TOPICS: \n")
# print(topics)

extractPhrases2(samples[0])
topics = extractTopics(samples[0])
print(topics)