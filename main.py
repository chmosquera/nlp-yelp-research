import pandas as pd
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from topic_extraction import topic_classification as TC
from sentiment_analysis import sentiment_analyzer as SA
nltk.download('sentiwordnet')
from datetime import datetime

def loadData(fileName='yelp-dataset/yelp_25k.csv'):
    global _df
    ## Import Dataset, restaurants only

    _df = pd.read_csv(fileName)
    restaurants = _df[_df['categories'].str.contains("Restaurants")]
    foods = _df[_df['categories'].str.contains("Food")]
    _df = pd.concat([restaurants, foods])

    return _df

def summarize(topic_scores):
    topics = {}
    for topic,score in topic_scores:
        if topic not in topics:
            topics[topic] = score
        else:
            topics[topic] += score

    for topic,score in topics.items():
        print(topic, ": ", score)

def log(lines):
    now = datetime.now()
    dt_string = now.strftime("-%d_%m_%Y-%H_%M")
    f = open("./logs/aspectSA.txt", 'w+')
    for line in lines:
        f.write(line + "\n")
    f.close()


def main():
    loadData()
    TC.loadData()

    log_data = []
    for text in _df['text'][:5]:
        log_data.append("----------------------------")
        log_data.append("Original text: ")
        log_data.append(text)
        log_data.append("----------------------------")
        raw_sentences = sent_tokenize(text.lower())
        topic_scores = []
        for sentence in raw_sentences:
            log_data.append(sentence)
            pairs = TC.aspectOpinionPair(sentence)
            score = SA.sentiWordNet(sentence, debug=False)
            for word, related_words,topic in pairs:
                topic_scores.append((topic,score))
                log_data.append("   (" + str(topic) + ") " + str(word) + " : " + str(round(score, 3)))
        summarize(topic_scores)
        log_data.append("\n")

    log(log_data)

# Global Vars
_df = None

if __name__ == "__main__":
    main()
