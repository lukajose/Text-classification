# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer

# %%
df=pd.read_csv("training.csv")
# bar graph of num docs per class
df2=df['topic'].value_counts()
df2.plot.barh(x="topic",y="count",title="Number of documents per topic (train set)",
 color=['black', 'red', 'green', 'blue', 'cyan','purple','yellow','orange','brown'])
plt.tight_layout()
plt.show()
plt.clf()


# %% [markdown]
# As shown above, data set is imbalanced. 
# We will need to use accuracy/loss measurements that take this into account during
# training (such as f1 score).

# %%
topics=[ c for c in df['topic'].unique()]
print(topics)

# %%
#bag of words - freq of words per topic
count = CountVectorizer()
for topic in topics:
    #get corpus per topic
    text_data=df.loc[(df['topic'] == topic)]
    text_data=text_data['article_words']

    bag_of_words = count.fit_transform(text_data)

    feature_array = np.array(count.get_feature_names())
    #sums each column so get an array of freq of each word across all docs. 
    #len(sum_words)= len(feature array)
    sum_words = np.array(bag_of_words.sum(axis=0)[0])

    word_freq={}
    for i in range(0, feature_array.shape[0]):
        word_freq[feature_array[i]]=sum_words[0,i]
    n=10
    sorted_word_freq=[(k,v) for k, v in sorted(word_freq.items(), reverse=True, key=lambda item:item[1])]
    top_n=[k[0] for k in sorted_word_freq]
    y=[k[1] for k in sorted_word_freq][:n]
    #print("most freq {}: {}".format(topic,top_n[:10]))

    #potentially remove these rare words from feature space...
    #print("least freq {}: {}".format(topic,top_n[-10:]))
    #plot bar chart of top words
    plt.bar(top_n[:n],y[:n])
    plt.title("Top words across category {}".format(topic))
    plt.tight_layout()
    plt.show()
    #plt.savefig('{}.jpg'.format(topic))
    plt.clf()


# %%

vectorizer = TfidfVectorizer()
#td without idf is a better measure than tdif according to general accuracy scores
vectorizer2 = TfidfVectorizer(use_idf=False)

for topic in topics:
    text_data=df.loc[(df['topic'] == topic)]
    text_data=text_data['article_words']

    ###from https://stackoverflow.com/questions/34232190/scikit-learn-tfidfvectorizer-how-to-get-top-n-terms-with-highest-tf-idf-score
    X = vectorizer2.fit_transform(text_data)
    print(X.shape)
    feature_array = np.array(vectorizer2.get_feature_names())
    sum_words = np.array(X.sum(axis=0)[0])	
    word_freq={}
    for i in range(0, feature_array.shape[0]):
        word_freq[feature_array[i]]=sum_words[0,i]
    n=10
    sorted_word_freq=[(k,v) for k, v in sorted(word_freq.items(), reverse=True, key=lambda item:item[1])]
    top_n=[k[0] for k in sorted_word_freq]
    y=[k[1] for k in sorted_word_freq]
    #print("most freq {}: {}".format(topic,top_n[:10]))

    #potentially remove these rare words from feature space...
    #print("least freq {}: {}".format(topic,top_n[-10:]))
    plt.bar(top_n[:n],y[:n])
    plt.title("Top tdf words across category {}".format(topic))
    plt.tight_layout()
    plt.show()
    #plt.savefig('tdf_only-{}.jpg'.format(topic))
    plt.clf()


# %%
#evauating accuracy of lr for bag of words, tdif, and tdf
lr=LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=1000,random_state=0)

#TD-IF
X = vectorizer.fit_transform(df['article_words'])	
X_train = X[:9400]
X_test = X[9400:]
y_train = df.iloc[:9400,2]
y_test = df.iloc[9400:,2]
s = cross_val_score(lr,X_train, y_train,cv=5)
print("mean accuracy of lr using tdif {}".format(s.mean()))

#wihtout idf- td  is just adjusting weighting by length of document. 
vectorizer2 = TfidfVectorizer(use_idf=False)
X = vectorizer2.fit_transform(df['article_words'])	
X_train = X[:9400]
X_test = X[9400:]

s = cross_val_score(lr,X_train, y_train, cv=5)
print("mean accuracy of lr using only td {}".format(s.mean()))


X = count.fit_transform(df['article_words'])	
X_train = X[:9400]
X_test = X[9400:]

s = cross_val_score(lr,X_train, y_train,cv=5)
print("mean accuracy of lr using bag of words {}".format(s.mean()))
lr.fit(X_train,y_train)


# %% [markdown]
#

# %%
#shuffle data
df = df.sample(frac=1).reset_index(drop=True)
#changing class weights to address the imbalance in dataset
lr=LogisticRegression(multi_class='multinomial',solver='lbfgs',class_weight='balanced',max_iter=1000,random_state=0)
#lets see accuracy using a validation set
print("bag of words")
X = count.fit_transform(df['article_words'])	
X_train = X[:9000]
X_test = X[9000:]
y_train = df.iloc[:9000,2]
y_test = df.iloc[9000:,2]
lr.fit(X_train,y_train)
predicted_y = lr.predict(X_test)
print(classification_report(y_test, predicted_y))


print("td")
X = vectorizer2.fit_transform(df['article_words'])	
X_train = X[:9000]
X_test = X[9000:]
y_train = df.iloc[:9000,2]
y_test = df.iloc[9000:,2]
lr.fit(X_train,y_train)
predicted_y = lr.predict(X_test)
print(classification_report(y_test, predicted_y))


print("TDIF")
X = vectorizer.fit_transform(df['article_words'])	
X_train = X[:9000]
X_test = X[9000:]
y_train = df.iloc[:9000,2]
y_test = df.iloc[9000:,2]
lr.fit(X_train,y_train)
predicted_y = lr.predict(X_test)
print(classification_report(y_test, predicted_y))

# %% [markdown]
# Some kind of modification is required to ensure the algorithm doesn't concentrate mainly 
# on those topics with most samples at the expense of accurately classifying other topics.
#
# Adding class weight ='balanced' to give small topics more weight, decreases overall accuracy, but increases accuracy for small topics by a lot (which is more important given the task of recommending articles per topic)
#

# %%

# %%

# %%

# %%

# %%

# %%

# %%
