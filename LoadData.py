#!/usr/bin/env python
# coding: utf-8

# In[2]:


import argparse

import pandas as pd
import numpy as np
import scipy
import re
import gensim

from collections import Counter

from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

parser = argparse.ArgumentParser(description='Load training and test data that we use in all the classification models.')

parser.add_argument('-t', '--test_size', type=float, required = False,
                    help='fraction of the data used for test (between 0 and 1)')

args = parser.parse_args()

if(args.test_size == None):
    TEST_SIZE = 0.5
else: 
    TEST_SIZE = args.test_size

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)  

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
def get_mean_w2v_embeddings(titles):
    embs = []
    for title in titles:
        title_emb = np.zeros(300)
        words = title.split(' ')
        for w in words:
            if w in word2vec_model:
                scalar = 1.
#                 scalar = 1. / len(words)
                
                vector = word2vec_model[w]
                
                title_emb += scalar * vector
        embs.append(title_emb)
    return embs

kicked = pd.read_csv('data/dismissed_complete.csv')
stayed = pd.read_csv('data/nodismissed_complete.csv')

kicked1 = kicked[['Author','Title paper', 'labels']]
stayed1 = stayed[['Author','Title paper', 'labels']]
stayed1 = stayed1.sample(frac=(1.0 * kicked.shape[0])/stayed.shape[0]) # random_state = 0

df0 = pd.concat([kicked1, stayed1])

df1 = df0.copy()
df1['Author'] = df0['Author'].apply(lambda x: x[3:])
df1['Label'] = df0['labels'].apply(lambda x: int(x))
df1['Title paper'] = df0['Title paper'].apply(lambda s: s[1:][:-1])
df1 = df1.drop(columns=['labels'])

df2 = df1[df1['Author'].apply(lambda s : len(s) >= 6)]

df3 = df2.copy()
df3['Title paper'] = df2['Title paper'].apply(
    lambda s : s.lower()
).apply(
    lambda s : re.sub(r"[.,/()?:'%\";\[\]!\{\}><]", "", s) # delete all not-letters
).apply(
    lambda s : re.sub(r"[- + = @ & * # |]", " ", s) # substitute defis with spaces
).apply(
    lambda s : re.sub(r"\d", " ", s) # substitute numbers with spaces
).apply(
    lambda s : re.sub(r"\W\w{1,2}\W", " ", s) # naive removal of super-short words
).apply(
    lambda s : re.sub(r"\s+", " ", s) # substitute multiple spaces with one
)
df3 = df3[df3['Title paper'].apply(
    lambda s: s != 'untitled' and s != 'editorial' # drop some common but not-interesting names
)]

# try to find strange symbols in "Title paper" and print them 
symbols = df3['Title paper'].apply(
    lambda s: ''.join(c for c in s if not c.isalpha() and c != ' ')
)

# okay, now in df3 in "Title paper" we have clean sentences, great, analysis should work

df4 = df3.drop(columns=['Author'])
df4.head()

titles_num = df4.shape[0]
kicked_titles_num = df4[df4['Label'] == 1].shape[0]
stayed_titles_num = df4[df4['Label'] == 0].shape[0]
print('For first dataset we have ' + str(titles_num) + ' titles, from them ' + str(kicked_titles_num) + ' kicked and ' + str(stayed_titles_num) + ' stayed')

# authors_num = len()

titles_per_author = {} # author -> article
labels_per_author = {} # author -> label

for i, r in df3.iterrows():
    author = r['Author']
    title = r['Title paper']
    label = int(r['Label'])
    
    titles_per_author[author] = titles_per_author.get(author, '') + ' ' + title # do concatenation
    labels_per_author[author] = label

kicked_cnt = 0
for k, v in labels_per_author.items():
    if v == 1: kicked_cnt+= 1
        
authors = []
titles = []
labels = []
stayed_limit = kicked_cnt

for k, v in titles_per_author.items():
    if labels_per_author[k] == 0:
        if stayed_limit > 0: stayed_limit -= 1
        else: continue
    
    authors.append(k)
    titles.append(re.sub(r"\s+", " ", v))    
    labels.append(labels_per_author[k])
    
# aggregated DataFrame
adf4 = pd.DataFrame(data={'Title paper' : titles, 'Label' : labels, 'Author': authors}) # columns names are ugly, but for backwards-compatibility


labels_per_titles = {}
same_duplicate = 0
diff_duplicate = 0

# we assume, that noone meets 3 times, which +- correct
for i, r in adf4.iterrows():
    title = r['Title paper']
    label = int(r['Label'])
    if title in labels_per_titles:
        if labels_per_titles[title] == label:
            same_duplicate += 1
        else:
            diff_duplicate += 1
    else:
        labels_per_titles[title] = label
        
        
# to clean the data, let's throw away all the duplicates at all, both same and diff, everyone, who meets > 1 times
counter = Counter(adf4['Title paper'])

adf5 = adf4[
    adf4['Title paper'].apply(
        lambda title: counter[title] == 1
    )
]

print('New dataset size after duplicates removal is ' + str(adf5.shape[0]))


df = adf5


data_train, data_test = train_test_split(df, test_size= TEST_SIZE) # random_state = 0

X_train = data_train['Title paper']
y_train = data_train['Label']

X_test = data_test['Title paper']
y_test = data_test['Label']

print('Train size: ' + str(X_train.shape[0]) + ' vs test size: ' + str(X_test.shape[0]))

X_train_embs = get_mean_w2v_embeddings(X_train)
X_test_embs  = get_mean_w2v_embeddings(X_test)

data_train['Title vector'] = pd.Series(X_train_embs, index=data_train.index)
data_test['Title vector'] = pd.Series(X_test_embs, index=data_test.index)

data_train.to_csv('data_train.csv')
data_train.to_csv('data_test.csv')





