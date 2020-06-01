
## Imports
import nltk, spacy, pickle, time
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
from pprint import pprint
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser

def loadData():
    global _df
    ## Import Dataset
    _df = pd.read_csv('yelp-dataset/yelp_25k.csv')
    print(_df.head())
    return _df


## Preprocessing text
stemmer = nltk.stem.porter.PorterStemmer() #NLTK's built-in stemmer resource
nlp = spacy.load('en', disable=['parser', 'ner']) # keep only tagger component (for efficiency)
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#uses NLTK's built in word tokenizer. Output is a list now.
def myTokenizer(text):
  return nltk.word_tokenize(text)

#POS tagger.
#input is a list of words, output is a list of tuples (Word, Tag)
def getPOS(tokenized):
  return nltk.pos_tag(tokenized)

#This removes all the stop words, and actually also punctuation and non-alphabetic words, and makes all lower case
#you can edit your own version
def filterTokens(tokenized):
  return [token.lower() for token in tokenized if token.isalpha() and token.lower() not in stopwords.words('english')]

#Using the NLTK stemmer
def stemming(tokenized):
  return [stemmer.stem(token) for token in tokenized]

def clean(sentences):
    start = time.time()
    for sentence in sentences:
        yield(filterTokens(myTokenizer(str(sentence))))  # deacc=True removes punctuations
    end = time.time()
    print("Time elapsed: ", end - start)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    start = time.time()
    result = [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stopwords.words('english')] for doc in texts]
    end = time.time()
    print("Time elapsed: ", end - start)
    return result

def make_bigrams(texts):
    start = time.time()
    bigrams = [bigram_model[doc] for doc in texts]
    end = time.time()
    print("Time elapsed: ", end - start)
    return bigrams


def loadBigramModel():
    bigram_model = Phraser.load(BIGRAM_MODEL_PATH)

# Do this the first time when you don't have any models
def buildModels():
    loadData() # _df
    # Data Example
    sample = _df.iloc[10]['text']
    print("original: ", sample)
    tokenized = myTokenizer(sample)
    print("tokenized: ", tokenized)
    filtered = filterTokens(tokenized)
    print("filtered: ", filtered)

    # Tokenize words and clean-up text
    sentences = _df.text
    data_words = list(clean(sentences))

    # ## Build bigram and trigram models
    # bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    # bigram_mod = gensim.models.phrases.Phraser(bigram)
    #
    # # Save / load an exported collocation model.
    # bigram_mod.save(BIGRAM_MODEL_PATH)

    ##
    data_words_nostops = remove_stopwords(data_words)
    # data_words_bigrams = make_bigrams(data_words_nostops)
    # data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'VERB', 'PNOUN'])

    ## Create dictionary and corpus needed for topic modeling
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]      # token2id : token -> tokenid; id2token : reverse mapping of token2id; doc2bow : convert doc to bag of words

    ## Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=20,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)

    lda_model.save(LDA_MODEL_PATH)

    ## Example This predicts the topics for a review
    predicted = lda_model[corpus[:5]]
    for (topic_id, topic_score) in predicted[0][0]:
        print(topic_id, ". ", topic_score, " : ", lda_model.show_topic(topic_id))


# TRIGRAM_MODEL_PATH = "./models/my_bigram_model.pkl"
BIGRAM_MODEL_PATH = "./models/my_bigram_model.pkl"
LDA_MODEL_PATH = "./models/my_lda_model.pkl"
if __name__ == "__main__":
    _df = None
    trigram_model = None
    bigram_model = None
