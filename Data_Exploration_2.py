# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix
from sklearn.svm import SVC
import random
sns.set()

train = pd.read_csv('training.csv',index_col="article_number")


train.head(10)

topics = train['topic'].unique()

topics = list(topics)

len(train) # number of observations

# # Data Analysis
# Visualizing document features, distributions of categories, and other stats

plt.figure(figsize=(10,3))
plt.legend()
train['topic'].value_counts().plot(kind='barh',color='green') # distribution of classes


# News article length per topic

# +

def fn(entry):
    return len(entry.split(","))
df=train
df['article_words'] = df['article_words'].astype(str)
df['Article_length'] = df['article_words'].apply(fn)
#print(df['Article_length'])

plt.figure(figsize=(15,8))
chart=sns.boxplot(data=df, x='topic', y='Article_length',color="#3498db")
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.xlabel("Topics")
plt.ylabel("Article length (number of words)")

# -

# ## Word Vectorizer
# Looking for patterns in data

# Get the word frequency of the categories
def get_word_frequency(corpus):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq
word_freq = get_word_frequency(train[train['topic']  == "IRRELEVANT"]['article_words'])
df1 = pd.DataFrame(word_freq, columns = ['ReviewText' , 'count'])
df1.head(5)



# ## Checking number of words
# This section plots the distribution of number of words per category

# Get categories
categories = sorted(list(train['topic'].unique()))
categories

vec = CountVectorizer().fit(train['article_words'])
bag_of_words = vec.transform(train['article_words'])
train['length_doc'] = bag_of_words.sum(axis=1)

# Plot distribution of words
plt.figure(figsize=(14,5))
for cat in categories:
    art_class = train[train['topic'] == cat]['length_doc']
    sns.distplot(art_class,label=cat,hist=False)
plt.xlabel("length of document")
plt.ylabel("kernel density ")
plt.legend();

# ## Proportion of classes
# Get the % of each class 

total = len(train)
for cat in categories:
    cat_num = len(train[train['topic'] == cat])
    print(f'Category: {cat} {cat_num/total}')


# ### Creating a word frequency top 5 words
# This graph will show a count frequency of the top 5 words per category.

topics = sorted(list(train['topic'].unique()),reverse=True)
topics

columns = 4
rows = 3
topics = sorted(list(train['topic'].unique()),reverse=True)
fig, axs = plt.subplots(rows,columns,figsize=(15,15))
fig.suptitle("Top 5 words frequency per category ",fontsize=20)
for col in range(columns):
    for row in range(rows):
        if len(topics) > 0:
            category = topics.pop()
            sample = train[train['topic'] == category]
            word_freq = get_word_frequency(sample['article_words'])
            df1 = pd.DataFrame(word_freq, columns = ['ReviewText' , 'count'])
            #print(df1.iloc[:5,:])
            axs[row][col].set_title(f"{category}")
            axs[row][col].bar(df1.iloc[:5,:]['ReviewText'],df1.iloc[:5,:]['count'])
axs[row][col].set_visible(False)

# ### Getting the chi2 unigram Tfidf
# Calculate the tfidf sparse matrix, get unigrams and bigrams using the chi2 to find most correlated words per category

#transform text to tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(train.article_words).toarray()
labels = train.topic
topics = sorted(list(train['topic'].unique()),reverse=False)
features.shape
print(topics)

#get top 5 most correlated words (unigrams) and bigrams per category
from sklearn.feature_selection import chi2
import numpy as np
N = 5
for Product, category_id in enumerate(topics):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  #unigrams = [(v,features_chi2[0][indices[i+=1]]) for v in feature_names if len(v.split(' ')) == 1]
  unigrams = []
  for c,v in enumerate(feature_names):
    add = (v,features_chi2[0][indices[c]])
    if len(v.split(' ')) == 1:
      unigrams.append(add)
        
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(category_id))
  print("  . Most correlated unigrams:")
  print(unigrams[-N:])

  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

df_unigram = pd.DataFrame(unigrams,columns=['words','chi2'])
df_unigram.iloc[-5:,:]

# Graphing most correlated unigrams
columns = 4
rows = 3
topics = sorted(list(train['topic'].unique()),reverse=True)
fig, axs = plt.subplots(rows,columns,figsize=(15,15))
fig.suptitle("Top 5 words TFIDF unigram chi2 per category",fontsize=20)
#initialize large sparse matrix
for col in range(columns):
    for row in range(rows):
        if len(topics) > 0:
            category = topics.pop()
            features_chi2 = chi2(features, labels == category)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(tfidf.get_feature_names())[indices]
            unigrams = []
            for c,v in enumerate(feature_names):
                add = (v,features_chi2[0][indices[c]])
                if len(v.split(' ')) == 1:
                  unigrams.append(add)
            df1 = pd.DataFrame(unigrams, columns = ['ReviewText' , 'tfidf-chi2'])
            #print(df1.iloc[:5,:])
            axs[row][col].set_title(f"{category}")
            axs[row][col].bar(df1.iloc[-5:,:]['ReviewText'],df1.iloc[-5:,:]['tfidf-chi2'],color="green")
axs[row][col].set_visible(False)

# ### Transform topic to numerical categorical data
#

from sklearn import preprocessing
"""used to convert categorical to id for predicting documents. sklearn trains different binary models based on id of target """
le = preprocessing.LabelEncoder() 
le.fit(train['topic'])
list(enumerate(le.classes_))

train['target'] = le.transform(train['topic'])

train[['topic','target']].head(10)

# ### Split data
# 9000 observations to train and 500 to use as development test set

X_train,X_test_dev = train.iloc[:9000,:-1],train.iloc[9000:,:-1]
y_train, y_test_dev = train.iloc[:9000,-1],train.iloc[9000:,-1]

X_train.tail(2)

X_test_dev.head(5)


#
# ## Using TFIDF as features
#
# TFIDF(t,d) = TF(t,d) x log(N /DF(t) )
#
# * t = word in document
#
# * d = document
#
# * N = number of documents in corpus
#
# * DF(t) = number of documents in the corpus containing frequency t.
#
# We will try to apply TFIDF without balancing the data and balancing the data to see if there are better results
#

class PipelineFeatureExtraction:
    def __init__(self,articles):
        self.articles = articles
        self.count = None
        self.tfidf = None
    def count_vector(self):
        count = CountVectorizer()
        text_data = self.articles.values #numpy array
        bag_of_words = count.fit_transform(text_data)
        self.count = count
        return bag_of_words
    def tfidf_vector(self):
        tfidf = TfidfVectorizer(sublinear_tf=True,max_features=5000)
        text_data = self.articles.values #numpy array
        tfidf_feat = tfidf.fit_transform(text_data)
        self.tfidf = tfidf
        return tfidf_feat



# # SVM + tfidf Not balancing the data

sv = SVC(kernel='linear',C=1)

# +
# exploring tfidf
#pipe.tfidf.vocabulary_
# -

pipe = PipelineFeatureExtraction(X_train['article_words'])
X_train_count = pipe.count_vector()
X_train_tfidf = pipe.tfidf_vector()

pipe.tfidf

sv.fit(X_train_tfidf,y_train)

X_train_tfidf.shape

y_train.shape

# creating test data
test_pipe = PipelineFeatureExtraction(X_test_dev['article_words'])
X_test_count = test_pipe.count_vector()
X_test_tfidf = test_pipe.tfidf_vector()

y_pred_train = sv.predict(X_train_tfidf)
y_pred_dev_test = sv.predict(X_test_tfidf)

X_test_tfidf.shape

# predictions training 
accuracy_score(y_train,y_pred_train)

#prediction test dev
accuracy_score(y_test_dev,y_pred_dev_test)

confusion_matrix(y_test_dev,y_pred_dev_test) #keeps classifying all documents as irrelevant

irrel = len(X_train[X_train['topic'] == "IRRELEVANT"])
money_mark = len(X_train[X_train['topic'] == "MONEY MARKETS"])
print('irrelevant:',len(X_train[X_train['topic'] == "IRRELEVANT"].index))
print('next_class_num:',len(X_train[X_train['topic'] == "MONEY MARKETS"].index)) # Lets reduce irrelevant at random to have the same number as money markets

# Balancing the data to next nearest class
toDrop = list(train[train['topic'] == "IRRELEVANT"].index)
l = len(toDrop) #total length of list
print(l)
resample_size = money_mark
for _ in range(resample_size):
    toDrop.pop(random.randrange(l))
    l = len(toDrop)

len(toDrop)

len(X_train)

len(y_train)

train.drop(toDrop,inplace=True)

len(train)

#Training set 
plt.figure(figsize=(10,3))
plt.legend()
train['topic'].value_counts().plot(kind='barh',color='blue') # distribution of classes

train.tail(10)

len(train[train['topic'] == "IRRELEVANT"])

X_train,X_test_dev = train.iloc[:5620,:-1],train.iloc[5620:,:-1]
y_train, y_test_dev = train.iloc[:5620,-1],train.iloc[5620:,-1]

pipe = PipelineFeatureExtraction(X_train['article_words'])
X_train_count = pipe.count_vector()
X_train_tfidf = pipe.tfidf_vector()

sv_balanced = SVC(kernel='linear')
sv_balanced.fit(X_train_tfidf,y_train)

pipe = PipelineFeatureExtraction(X_test_dev['article_words'])
X_test_count = pipe.count_vector()
X_test_tfidf = pipe.tfidf_vector()

y_pred_train = sv_balanced.predict(X_train_tfidf)
accuracy_score(y_train,y_pred_train)

y_pred_test = sv_balanced.predict(X_test_tfidf)
accuracy_score(y_test_dev,y_pred_test)

confusion_matrix(y_test_dev,y_pred_test)

len(toDrop)

# # SGD SVM Irrelevant + classes

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
train = pd.read_csv('training.csv',index_col="article_number")
sgd = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', SGDClassifier(loss='hinge')),
              ])


le = preprocessing.LabelEncoder() 
le.fit(train['topic'])
classes = list(le.classes_)
train['target'] = le.transform(train['topic'])
X_train,X_test_dev = train.iloc[:9000,:-1],train.iloc[9000:,:-1]
y_train, y_test_dev = train.iloc[:9000,-1],train.iloc[9000:,-1]


print(sorted(list(y_train.unique())))
print(classes)

sgd.fit(X_train['article_words'],y_train)
y_pred_test = sgd.predict(X_test_dev['article_words'])
print(f'accuracy {accuracy_score(y_test_dev,y_pred)}')
print(classification_report(y_test_dev, y_pred,target_names=classes))

confusion_matrix(y_test_dev,y_pred_test)

test = pd.read_csv('test.csv')
y_test = le.transform(test['topic'])
X_test = test['article_words']
y_pred = sgd.predict(X_test)
print(f'accuracy {accuracy_score(y_test,y_pred)}')
print(classification_report(y_test, y_pred,target_names=classes))
