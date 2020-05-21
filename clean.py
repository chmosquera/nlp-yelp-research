import pandas as pd
import json

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



df = pd.read_csv('yelp-dataset/yelp_25k.csv')
getRestaurantBusinessData()