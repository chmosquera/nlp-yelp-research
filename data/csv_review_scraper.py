import pandas as pd

df = pd.read_csv("good_reviews.csv")
print(df['text'][:10])

output = "scraped_reviews/"
reviews = list(df['text'])
for i in range(len(reviews)):
	f = open(output + "good_review_" + str(i) + ".txt", "w")
	f.writelines(reviews[i])
	f.close()


