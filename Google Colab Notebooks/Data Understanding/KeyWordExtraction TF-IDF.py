import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer

import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
#

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:

        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results


pd.set_option('display.max_columns', None)
df = pd.read_csv("all_songs_preprocessed.csv")

df = df.loc[:, ~df.columns.str.contains('Unnamed')]

lyrics = df['Corpus'].tolist()
print (df.head())
print (len(lyrics))


stop_words = stopwords.words('english')
print (stop_words)
cv = CountVectorizer(max_df = 0.85,stop_words=stop_words, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")#3indicates the min size

word_count_vector = cv.fit_transform(lyrics)
print (word_count_vector)
print (list(cv.vocabulary_.keys())[:10])#List ten keys
# print (word_count_vector)
print (cv.get_feature_names())

##Idea fit the data to the whole dataset
#Transform only one document at once and save top n vectors
tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
tf_idf_vector = tfidf_transformer.fit_transform(word_count_vector)

print (tf_idf_vector)
sorted_items = sort_coo(tf_idf_vector.tocoo())

print (sorted_items)

feature_names = cv.get_feature_names()


keywords = extract_topn_from_vector(feature_names,sorted_items,100)

print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])


##Tracking down why some keywords might occur
print (df.Corpus.str.count("seven").max())
print(df.song_title.iloc[df.Corpus.str.count("seven").argmax()])
print (df.Corpus.str.count("Kanye").sort_values())
print (df.Corpus.iloc[10063])



































##
