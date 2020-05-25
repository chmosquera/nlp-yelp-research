#Don't need these anymore
import pandas as pd

df_business = pd.read_csv('yelp-dataset/yelp_business.csv')

all_categories = []
for line in df_business.categories:
    tokens = line.split(';')
    for tok in tokens:
        if tok not in all_categories:
            all_categories.append(tok)

f = open("categories.txt", 'w')
f.write('\n'.join(all_categories))
f.close()