import pandas as pd

def loadData(fileName='D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\data\\human.csv'):
    # Import Dataset, restaurants only

    _df = pd.read_csv(fileName)

    return _df


def calculateAESO():
    # Prepare dataframe to save data
    df = {'review_id':[], 'text':[], 'food_score':[], 'atms_score':[], 'serv_score':[], 'prce_score':[]}

    human_scores = loadData()

    for idx,row in human_scores.iterrows():
        ex = (row['review_id'], row['text'])
        df['text'].append(row['text'])
        df['review_id'].append(row['review_id'])

        # ########################
        # Example of pipeline
        # ########################
        LOGDATA = []
        LOGDATA.append("###########################################################")
        LOGDATA.append("review:: " + condenseReview(ex[1]))
        LOGDATA.append("###########################################################")
        TT = PrettyTable(["sentence", "subj", "sent_score", "topic", "sim_score"])

        nouns_sentiments = aspectSentimentCalculation(ex[1])
        nouns = [s for s,senti,sentence in nouns_sentiments]
        assigned_topics = yelpSimilarities(ex[1],nouns)

        temp_join = list(zip(nouns_sentiments, assigned_topics))
        subj_topic_sentiment = []   # (phrase, subject, sentiment, assigned topic, similarity score)
        summary_tup = []                # (topic, sentiment)
        for ns,at in temp_join:
            if not ns[0] == at[0]:
                print("Error Mismatch: Trying to join subject/sentiment with subject/topic. Subject '" + ns[0] + "' does not match '" + at[0] + "'")
                continue
            # add row: phrase, subject, sentiment, assigned topic, similarity score
            subj_topic_sentiment.append((ns[2], ns[0], ns[1], at[1][0], at[1][1]))
            summary_tup.append((at[1][0], ns[1]))
            TT.add_row([ns[2], ns[0], ns[1], at[1][0], at[1][1]])

        summary = summarizeResults(summary_tup)

        df['food_score'].append(summary[0][1])
        df['atms_score'].append(summary[1][1])
        df['serv_score'].append(summary[2][1])
        df['prce_score'].append(summary[3][1])

        # LOGDATA.append(topic + " (" + score + ") " for topic,score in summary)
        # LOGDATA.append(TT.get_string())

        # # save file
        # save = open("complete_analysis.txt", 'a+')
        # save.writelines('\n'.join(LOGDATA) + '\n')
        # save.close()


    # Save to csv
    system_scores = pd.DataFrame(df)
    system_scores.to_csv('D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\data\\results.csv')



##
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def norm(score):
    if score == 0:
        return 0
    elif score > 0:
        return 1
    elif score < 0:
        return -1

def accuracy_one(x_summary,y_summary):
  x_vec = np.array([tup[1] for tup in x_summary]).reshape(1,4)
  y_vec = np.array([tup[1] for tup in y_summary]).reshape(1,4)

  dist = euclidean_distances(x_vec, y_vec)
  print("dist: distance")
  norm_dist = dist/4.0    # max distance is 4.0

  return (1- norm_dist)

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

    return sum(accuracy)/len(accuracy)


##
fileName = "D:\\OneDrive - California Polytechnic State University\\csc582\\yelp final project\\nltk-yelp-research\data\\human.csv"
test_data = pd.read_csv(fileName)

# Perform aspect sentiment analysis on the data set
system_df = calculateAESO(test_data)

if not len(test_data) == len(system_df):
    print("Error: Data shapes are not the same. " + str(len(test_data)) + "!=" + str(len(system_df)))

x_summaries = []
y_summaries = []
row_count = len(test_data)
for i in range(row_count):
    x_row = test_data.iloc[i]
    y_row = system_df.iloc[i]
    x_summaries.append([('FOOD', norm(float(x_row['food_score']))), ('ATMS', norm(float(x_row['atms_score']))), ('SERV', norm(float(x_row['serv_score']))), ('PRCE', norm(float(x_row['prce_score'])))])
    y_summaries.append([('FOOD', norm(float(y_row['food_score']))), ('ATMS', norm(float(y_row['atms_score']))), ('SERV', norm(float(y_row['serv_score']))), ('PRCE', norm(float(y_row['prce_score'])))])

score = accuracy_metric(x_summaries, y_summaries)
print(score)


##
aa = [('FOOD', 1), ('ATMS', 0), ('SERV', 1), ('PRCE', 0)]
bb = [('FOOD', 1), ('ATMS', 1), ('SERV', 0), ('PRCE', 0)]
cc = [('FOOD', 1), ('ATMS', 1), ('SERV', 0), ('PRCE', 0)]
dd = [('FOOD', 1), ('ATMS', 0), ('SERV', 0), ('PRCE', 0)]

x = [aa, bb, cc]
y = [cc, bb, dd]

score = accuracy_metric(x, y)
print(score)