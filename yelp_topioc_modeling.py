
## Imports
import nltk, spacy, pickle, time
from nltk.corpus import stopwords
from datetime import datetime
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
from gensim.models.ldamodel import LdaModel
from apyori import apriori


def loadData(fileName = None):
    global _df
    ## Import Dataset, restaurants only
    if fileName is None:
        fileName = 'yelp-dataset/' + DF_NAME + '.csv'

    _df = pd.read_csv(fileName)
    restaurants = _df[_df['categories'].str.contains("Restaurants")]
    foods = _df[_df['categories'].str.contains("Food")]
    _df = pd.concat([restaurants, foods])
    print(_df.head())

    return _df

## Preprocessing text
stemmer = nltk.stem.porter.PorterStemmer() #NLTK's built-in stemmer resource
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # keep only tagger component (for efficiency)
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#uses NLTK's built in word tokenizer. Output is a list now.
def tokenize(raw):
  return nltk.word_tokenize(raw)

def tokenizeAll(raw_texts):
    start = time.time()
    tokenized = []
    for text in raw_texts:
        tokenized.append(tokenize(str(text)))  # deacc=True removes punctuations
    end = time.time()
    print("Time elapsed: ", end - start)
    return tokenized

#POS tagger.
#input is a list of words, output is a list of tuples (Word, Tag)
def getPOS(tokenized):
  return nltk.pos_tag(tokenized)

#This removes all the stop words, and actually also punctuation and non-alphabetic words, and makes all lower case
#you can edit your own version
def filterTokens(tokenized):
    start = time.time()
    filtered = []
    for token in tokenized:
        if token.isalpha() and token.lower() not in stopwords.words('english'):
            filtered.append(token.lower())  # deacc=True removes punctuations
    end = time.time()
    print("Time elapsed: ", end - start)
    return filtered

def filterTokensAll(tokenized):
  return [token.lower() for token in tokenized if token.isalpha() and token.lower() not in stopwords.words('english')]

#Using the NLTK stemmer
def stemming(tokenized):
  return [stemmer.stem(token) for token in tokenized]

def clean(sentences):
    start = time.time()
    for sentence in sentences:
        yield(filterTokens(tokenize(str(sentence))))  # deacc=True removes punctuations
    end = time.time()
    print("Time elapsed: ", end - start)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def lemmatize(tokenized, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    tagged = nlp(" ".join(tokenized))
    return [token.lemma_ for token in tagged if token .pos_ in allowed_postags]

def lemmatizeAll(tokenized_text):
    print("Lemmatizing ", len(tokenized_text), "rows")
    start = time.time()
    lemmatized = []
    for tokenized in tokenized_text:
        lemmatized.append(lemmatize(tokenized))  # deacc=True removes punctuations
    end = time.time()
    print("     Time elapsed: ", end - start)
    return lemmatized


## Define functions for stopwords, bigrams, trigrams and lemmatization
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
def extractPhrases(lemmatized):
    text = " ".join(lemmatized)
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    tree = regexChunker(tagged)
    phrases = []
    for subtree in tree.subtrees():
        if subtree.label() in ["NP", "PP", "VP", "CLAUSE", "NOUN"]:
            p = " ".join([tag[0] for tag in subtree.leaves()])
            # print(p, subtree.label())
            phrases.append((p,subtree.label()))
    return phrases

def extractPhrasesAll(lemmatized_text):
    print("Lemmatizing ", len(lemmatized_text), "rows")
    start = time.time()
    phrases = []
    for lemmatized in lemmatized_text:
        np = extractPhrases(lemmatized)
        # print("np: ", np)
        n = extractNouns(lemmatized)
        # print("n: ", n)
        nnp = []
        nnp.extend(np)
        nnp.extend(n)
        phrases.append(nnp)
    end = time.time()

    # Also get single nouns
    print("     Time elapsed: ", end - start)
    return phrases


def extractNouns(lemmatized):
    NOUNS = ["NN","NNS","NNP","NNPS"]
    text = " ".join(lemmatized)
    tagged = nltk.pos_tag(nltk.word_tokenize(text))
    entities = []
    for word,tag in tagged:
        if tag in NOUNS:
            entities.append((word,tag))
    return entities

def loadBigramModel():
    bigram_model = Phraser.load(BIGRAM_MODEL_PATH)

def log(samples, dt_string, model, corpus, elapsed_time):
    f = open("logs\\topic_modeling.log", 'a+')
    f.write("\n===========================================================")
    # f.write("\n* No noun phrases, just nouns")
    f.write("\nData path: " + PICKLE_PATH + DF_NAME + dt_string + ".csv")
    f.write("\nnum_rows: " + str(len(samples)))
    f.write("\nModel path: " + LDA_MODEL_PATH + "my_lda_model" + dt_string + ".pkl")
    f.write('\nPerplexity: ' + str(model.log_perplexity(corpus)))  # a measure of how good the model is. lower the better.
    f.write("\nchunksize: " + str(model.chunksize))
    f.write("\nnum_topics: " + str(model.num_topics))
    f.write("\npasses: " + str(model.passes))
    f.write("\niterations: " + str(model.iterations))
    f.write("\nelapsed time: " + str(elapsed_time))
    f.write("\n===========================================================")
    topics = model.print_topics()
    for topic in topics:
        f.write("\n" + str(topic[0]) + ": ")
        f.write("\n" + str(topic[1]) + ": ")
    # f.write(str(model.print_topics()))
    f.close()


## Do this the first time when you don't have any models
def buildModels(log_info = True):
    # capture date/time, use for naming later
    now = datetime.now()
    dt_string = now.strftime("-%d_%m_%Y-%H_%M")

    loadData() # _df
    # # Data Example
    # sample = _df.iloc[100]['text']
    # print("original: ", sample)
    # tokenized = tokenize(sample)
    # print("tokenized: ", tokenized)
    # filtered = filterTokens(tokenized)
    # print("filtered: ", filtered)

    # Tokenize words and clean-up text
    samples = _df.text
    business_ids = _df.business_id
    tokenized = tokenizeAll(samples)
    lemmatized = lemmatizeAll(tokenized)
    phrase_pairs = extractPhrasesAll(lemmatized)

    phrases = [[phrase.lower() for phrase,label in pair] for pair in phrase_pairs]

    # Save cleaned texts
    data = {'business_id': business_ids,
            'text': samples,
            'tokens': tokenized,
            'lemma': lemmatized,
            'phrase_pairs': phrase_pairs}
    df_preprocessed = pd.DataFrame(data, columns=data.keys())
    df_preprocessed.to_pickle(PICKLE_PATH + DF_NAME + dt_string + ".pkl")
    df_preprocessed.to_csv(PICKLE_PATH + DF_NAME + dt_string + ".csv")


    id2word = corpora.Dictionary(phrases)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in phrases]      # token2id : token -> tokenid; id2token : reverse mapping of token2id; doc2bow : convert doc to bag of words

    ## Build LDA model
    print("Building model...")
    num_topics = []
    start = time.time()
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=25,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=1000,
                                                passes=20,
                                                iterations=500,
                                                alpha='auto',
                                                eval_every=None,
                                                per_word_topics=True)
    end = time.time()
    print("     Time elapsed: ", end-start)

    # Save the LDA model. Using very specific name so I can save all models
    lda_model.save(LDA_MODEL_PATH + "my_lda_model" + dt_string + ".pkl")

    pprint(lda_model.print_topics())
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Log information so I can refer back to it later.
    if log_info:
        log(samples, dt_string, lda_model, corpus, end-start)

    # This is taking tooooooo long to do just 10 rows
    # # Build apriori
    # association_rules = apriori(phrases, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
    # association_results = list(association_rules)


def buidModelOnly():

    # load data
    loadData()


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


def applyLDAModel():

    loadData() # _df

    # Tokenize words and clean-up text
    samples = _df.text
    business_ids = _df.business_id
    tokenized = tokenizeAll(samples)
    lemmatized = lemmatizeAll(tokenized)
    phrase_pairs = extractPhrasesAll(lemmatized)

    phrases = [[phrase.lower() for phrase,label in pair] for pair in phrase_pairs]

    id2word = corpora.Dictionary(phrases)
    corpus = [id2word.doc2bow(text) for text in phrases]      # token2id : token -> tokenid; id2token : reverse mapping of token2id; doc2bow : convert doc to bag of words

    df_topic_sents_keywords = format_topics_sentences(ldamodel=_lda_model, corpus=corpus, texts=samples)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show
    df_dominant_topic.head(10)
    df_dominant_topic.to_csv("lda_results.csv")


# def applyLDAModel():
#     fileName = 'pickles/yelp_25k-01_06_2020-10_01.csv'
#     data = pd.read_csv(fileName)
#
#     # Tokenize words and clean-up text
#     samples = data.text
#     tokenized = data.tokens
#     lemmatized = data.lemma
#     phrase_pairs = data.phrase_pairs
#
#     phrases = [[phrase.lower() for phrase,label in pair] for pair in phrase_pairs]
#
#     id2word = corpora.Dictionary(phrases)
#     corpus = [id2word.doc2bow(text) for text in phrases]      # token2id : token -> tokenid; id2token : reverse mapping of token2id; doc2bow : convert doc to bag of words
#
#     df_topic_sents_keywords = format_topics_sentences(ldamodel=_lda_model, corpus=corpus, texts=samples)
#
#     # Format
#     df_dominant_topic = df_topic_sents_keywords.reset_index()
#     df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
#
#     # Show
#     df_dominant_topic.head(10)
#     df_dominant_topic.to_csv("lda_results.csv")

# TRIGRAM_MODEL_PATH = "./models/my_bigram_model.pkl"
BIGRAM_MODEL_PATH = "./models/my_bigram_model.pkl"
LDA_MODEL_PATH = "./models/"
DF_NAME = "yelp_25k"
PICKLE_PATH = "pickles\\"

if __name__ == "__main__":
    _df = None
    bigram_model = None
    _lda_model = LdaModel.load("./models/my_lda_model-01_06_2020-10_01.pkl")


# buildModels()

applyLDAModel()