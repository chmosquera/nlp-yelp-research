import testing_topic_classification as TC
import testing_sentiment_analysis as SA
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

def average(sum,cnt):
    if cnt == 0:
        return 0
    return sum / cnt
        

def summarizeScores(topic_senti, category=['food', 'atms', 'serv', 'prce']):
    scores = {cat:0 for cat in category}
    counts = {cat:0 for cat in category}
    avg = {cat:0 for cat in category}

    for topic,score in topic_senti:
        if topic not in category:
            print("Error: Missing topic from list of categories. Topic='" + topic + '"')
            exit()
        scores[topic] += score
        counts[topic] += 1

    avg_senti_per_topic = {cat:average(scores[cat],counts[cat]) for cat in category}
    return avg_senti_per_topic

def topicSentiment(review, soft=False, category=['food', 'atms', 'serv', 'prce']):

    # 1 split review into sentences
    sentences = sent_tokenize(review.lower())

    # 2 Classify the topic of each sentence, and analyze its sentiment
    topic_sentiments = []
    for sentence in sentences:
        # i classify topic
        most_similar_topics = TC.topicClassification(sentence, TC.seeds, use_s2v=True)
        topic = list(most_similar_topics.keys())[0]

        # ii calculate sentiment
        senti = SA.sentimentAnalysis(sentence, use_bing=True, soft=soft)

        topic_sentiments.append((topic, senti))
        print(topic, senti, sentence)

    # summarize the topic/senti
    summary = summarizeScores(topic_sentiments)    
    return summary
