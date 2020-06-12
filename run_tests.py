import testing_topic_classification as TC
import testing_sentiment_analysis as SA
import testing_topic_sentiment_analysis as TCSA

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


##########################################################################
# Tools
##########################################################################

def norm(score):
    if score == 0:
        return 0
    elif score > 0:
        return 1
    elif score < 0:
        return -1


def accuracy_one(actual, predicted):
    x_vec = np.array([vec for vec in actual]).reshape(1, 4)
    y_vec = np.array([vec for vec in predicted]).reshape(1, 4)

    dist = euclidean_distances(x_vec, y_vec)
    norm_dist = dist / 4.0  # max distance is 4.0

    return (1 - norm_dist)


def accuracy_metric(x_summaries, y_summaries):
    if not len(x_summaries) == len(y_summaries):
        print("Error: different shapes: " + len(x_summaries) + "!=" + len(y_summaries))
        return None

    accuracy = []
    num_points = len(x_summaries)
    for i in range(num_points):
        print(x_summaries[i], y_summaries[i])
        score = accuracy_one(x_summaries[i], y_summaries[i])
        print("accuracy: ", score)
        accuracy.append(score)

    return sum(accuracy) / len(accuracy), accuracy


##########################################################################
# INPUT CSV FILE
##########################################################################

file_name = 'combined_reviews'


##########################################################################
# Topic Classification
##########################################################################
def test_topic_classification(use_s2v, out_extension='_results_tc'):
    def vectorize_dict(cat_dict, category=['food', 'atms', 'serv', 'prce']):
        vec = []
        for cat in category:
            vec.append(float(cat_dict[cat]))
        return vec

    file = 'D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\\data\\' + file_name + '.csv'
    human_scores_df = pd.read_csv(file)

    # combine category rows from dataset into a dictionary
    categories = []
    for idx, row in human_scores_df.iterrows():
        cat_dict = {
            'food': row['food'],
            'atms': row['atms'],
            'serv': row['serv'],
            'prce': row['prce']
        }
        categories.append(cat_dict)

    # data
    x = human_scores_df['text']

    # Perform classifiaction
    predicted = [TC.topicClassification(p, TC.seeds, use_s2v=use_s2v) for p in x]

    # vectorize labels
    y = [vectorize_dict(row) for row in categories]
    predicted = [vectorize_dict(row) for row in predicted]

    # Calculate accuracy
    accuracy, accuracy_matrx = accuracy_metric(y, predicted)

    # save predicted to csv
    human_scores_df['manual_cats'] = y
    human_scores_df['predicted_cats'] = predicted
    human_scores_df['similarity'] = accuracy_matrx
    system_scores = pd.DataFrame(human_scores_df)
    out_file_name = file_name + out_extension + "_s2v(" + str(use_s2v) + ")"
    system_scores.to_csv(
        'D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\\data\\' + out_file_name + '.csv')

    return accuracy

##########################################################################
# Sentiment Analysis
##########################################################################
def test_sentiment_analysis(use_bing, use_lesk, out_extension='_results_sa'):

    def avg_sentiment_score(scores):
        return sum(scores) / len(scores)

    def normalize_sentiment_score(scores):
        norm = []
        for score in scores:
            if score > 0:
                norm.append(1)
            elif score < 0:
                norm.append(-1)
            else:
                norm.append(0)
        return norm

    # Load Hu/Liu's sentiment lexicon
    SA.loadSentimentLexicons()

    file = 'D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\\data\\' + file_name + '.csv'
    human_scores_df = pd.read_csv(file)

    # combine category rows from dataset into a dictionary, avg, then normalize -1 --> 1
    y = [avg_sentiment_score([row['food_score'], row['atms_score'], row['serv_score'], row['prce_score']]) for idx, row
         in human_scores_df.iterrows()]
    y = normalize_sentiment_score(y)

    # run through my sentiment algorithm
    x = human_scores_df['text']
    if use_lesk:
        predicted = [SA.sentimentAnalysisLesk(p, use_bing=use_bing, soft=False) for p in x]
    else:
        predicted = [SA.sentimentAnalysis(p, use_bing=use_bing, soft=False) for p in x]

    # calculate the accuracy
    if len(y) != len(predicted):
        print("Error")
    length = len(y)
    x_vec = np.array(y).reshape(1, length)
    y_vec = np.array(predicted).reshape(1, length)

    dist = euclidean_distances(x_vec, y_vec)
    print("dist: distance")
    norm_dist = dist / length  # max distance is 4.0

    accuracy = 1 - norm_dist
    print(accuracy)


    # accuracy, accuracy_matrx = accuracy_metric(y, predicted)

    human_scores_df['man_avg_score'] = y
    human_scores_df['pred_avg_score'] = predicted
    system_scores = pd.DataFrame(human_scores_df)
    out_file_name = file_name + out_extension + "_bing(" + str(use_bing) + ")_lesk(" + str(use_lesk)
    system_scores.to_csv(
        'D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\\data\\' + out_file_name + '.csv')

    return accuracy


def test_sentiment_analysis_socalc():
    def normalize_sentiment_score(scores):
        norm = []
        for score in scores:
            if score > 0:
                norm.append(1)
            elif score < 0:
                norm.append(-1)
            else:
                norm.append(0)
        return norm

    # Load Hu/Liu's sentiment lexicon
    SA.loadSentimentLexicons()

    # Get reviews from csv
    file = 'D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\\data\\combined_reviews_socalc.csv'
    reviews_df = pd.read_csv(file)

    # get so-calc scores
    y = normalize_sentiment_score(reviews_df['so-calc scores'])

    # run through my sentiment algorithm
    x = reviews_df['text']
    predicted = [SA.sentimentAnalysis(p, use_bing=True, soft=False) for p in x]

    # calculate the accuracy
    if len(y) != len(predicted):
        print("Error")

    length = len(y)
    x_vec = np.array(y).reshape(1, length)
    y_vec = np.array(predicted).reshape(1, length)

    dist = euclidean_distances(x_vec, y_vec)    
    print("dist: distance")
    norm_dist = dist / length  # max distance is 4.0

    accuracy =  1 - norm_dist
    print(accuracy)

    return accuracy

##########################################################################
# Topic-Sentiment Classifcation and Analysis
##########################################################################


def test_topic_sentiment(out_extension='_results_tcsa'):
    def vectorize_dict(cat_dict, category=['food', 'atms', 'serv', 'prce']):
        vec = []
        for cat in category:
            vec.append(float(cat_dict[cat]))
        return vec

    # load hu/liu's sentiment lexicon
    SA.loadSentimentLexicons()

    # load csv 
    file = 'D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\\data\\' + file_name + '.csv'
    human_scores_df = pd.read_csv(file)

    # combine topics-sentiment scores per row from dataset into a dictionary
    y = []
    for idx, row in human_scores_df.iterrows():
        cat_dict = {
            'food': row['food_score'],
            'atms': row['atms_score'],
            'serv': row['serv_score'],
            'prce': row['prce_score']
        }
        y.append(cat_dict)

    # data
    x = human_scores_df['text']
    predicted = [TCSA.topicSentiment(p, soft=False) for p in x]

    # vectorize labels
    y = [vectorize_dict(row) for row in y]
    predicted = [vectorize_dict(row) for row in predicted]

    # Calculate accuracy    
    accuracy, accuracy_matrx = accuracy_metric(y, predicted)

    # save predicted to csv
    human_scores_df['manual_cats'] = y
    human_scores_df['predicted_cats'] = predicted
    human_scores_df['similarity'] = accuracy_matrx
    system_scores = pd.DataFrame(human_scores_df)
    system_scores.to_csv(
        'D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\\data\\' + file_name + out_extension + '.csv')

    return accuracy


##########################################################################
# Run Tests
##########################################################################

f = open("D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\\data\\results.txt", 'a+')
# f.write("topic_classification: "+ str(test_topic_classification(use_s2v=False)) + "\n")
# f.write("topic_classification_s2v: "+ str(test_topic_classification(use_s2v=True)) + "\n")
# f.write("sentiment_analysis: "+ str(test_sentiment_analysis(use_bing=False, use_lesk=False)) + "\n")
f.write("sentiment_analysis_lesk: "+ str(test_sentiment_analysis(use_bing=False, use_lesk=True)) + "\n")
# f.write("sentiment_analysis_lesk_bing: "+ str(test_sentiment_analysis(use_bing=True, use_lesk=True)) + "\n")
# f.write("sentiment_analysis_vs_socalc: "+ str(test_sentiment_analysis_socalc()) + "\n")
# f.write("topic_sentiment: "+ str(test_topic_sentiment()) + "\n")
f.close()
