import pandas as pd
import json
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


def joinReviewsBusiness(row_cnt):
    output = "yelp_" + str(row_cnt) + ".csv"
    df_reviews = pd.read_csv('yelp-dataset/yelp_25k.csv',nrows=row_cnt)
    df_business = pd.read_csv('yelp-dataset/yelp_business_restaurant_temp.csv')
    result = pd.merge(df_reviews, df_business[['stars', 'review_count', 'categories', 'business_id']], how='right', on='business_id', suffixes=('','_business'))
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
    output = "yelp-dataset/yelp_business_restaurant_temp.csv"
    df_business = pd.read_csv('yelp-dataset/yelp_business.csv',nrows=10)

    isFood = False
    for idx, row in df_business.iterrows():
        print(idx,row.name, row.categories)
        if "Restaurants" not in row.categories and "Food" not in row.categories:
            print("dropped: ", idx)
            df_business = df_business.drop(idx)

    df_business.to_csv(output, sep=',', index=False)

# getRestaurantBusinessData()

joinReviewsBusiness(10)