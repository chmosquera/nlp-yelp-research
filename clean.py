import pandas as pd
import json
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


def joinReviewsBusiness(row_cnt):
    output = "yelp_" + row_cnt + ".csv"
    df_reviews = pd.read_csv('yelp-dataset/yelp_review.csv',nrows=row_cnt)
    df_business = pd.read_csv('yelp-dataset/yelp_business.csv')
    result = pd.merge(df_reviews, df_business[['stars', 'review_count', 'categories', 'business_id']], how='left', on='business_id', suffixes=('','_business'))
    result.to_csv(output, sep=',', index=False)
    df = pd.read_csv(output)
    print(len(df))

def removeNonFood(data):
    f = open('categories.txt', 'r')
    print(categories)
    s = [r.split(';') for r in data['categories']]
    print(s)

    isFood = False
    for index, row in data.iterrows():
        for c in row.categories:
            if c in categories:
                  isFood = True
        print(index, row)
        if not isFood:
            print("dropped: ", row)
            data.drop([index])

def getRestaurantBusinessData():
    output = "yelp_business_restaurant.csv"
    df_business = pd.read_csv('yelp-dataset/yelp_business.csv')

    isFood = False
    for index, row in df_business.iterrows():
        print(index,row.name, row.categories)
        if "Restaurants" not in row.categories:
            print("dropped: ", index)
            df_business.drop([index])

    df_business.to_csv(output, sep=',', index=False)



# df = pd.read_csv('yelp-dataset/yelp_25k.csv')
# getRestaurantBusinessData()

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

###T TODO: I was trying to create a function that takes the table of reviews and forms a new tables with
### TODO: the original review, tokenized, lemmatized, bigrams, etc. Just because it takes a while to do this on 25k reviews
# # Define functions for stopwords, bigrams, trigrams and lemmatization
# def remove_stopwords(texts):
#     return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stopwords.words('english')] for doc in texts]
#
# def make_bigrams(texts):
#     return [bigram_mod[doc] for doc in texts]
# def clean_to_table(csv_file, out_file, field_to_clean, fields_to_keep = ['id', 'text']):
#     df = pd.read_csv(csv_file)
#     sentences = df[field_to_clean]
#     data_words = [yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) for sentence in sentences]
#     data_words_nostops = remove_stopwords(data_words)
#     data_words_bigrams = make_bigrams(data_words_nostops)
#     data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#     new_df.to_csv(out_file, sep=',', index=False)