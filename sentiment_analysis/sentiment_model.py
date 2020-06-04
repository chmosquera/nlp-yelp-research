import pandas as pd
import random

data = pd.read_csv("./Reviews.csv")
data = data.sample(frac=1)[:20000]

data.columns = map(lambda x:x.lower(), list(data))
data["text"] = data["summary"] + " "+ data["text"]
data = data[["text", "score"]]

data.loc[data.score<3, "score"] = -1
data.loc[data.score==3, "score"] = 0
data.loc[data.score>3, "score"] = 1

data.head(5)

sentiment_data = zip(data["text"], data["score"])
random.shuffle(sentiment_data)

# 80% for training
train_X, train_y = zip(*sentiment_data[:16000])

# Keep 20% for testing
test_X, test_y = zip(*sentiment_data[16000:])