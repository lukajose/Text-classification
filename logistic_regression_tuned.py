# ---
# jupyter:
#   jupytext:
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

# %% [markdown]
# # Text Classification

# %% [markdown]
# ## Setup


# %%
# Importing Libraries
import hyperopt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import fmin, hp, space_eval, tpe
from sklearn.ensemble import VotingClassifier
from scipy.sparse import csr_matrix
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
   # plot_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
# Change default plot size
matplotlib.rcParams["figure.figsize"] = (12, 8)

# Setting random seed
np.random.seed(42)

# %%
# Importing data
train = pd.read_csv("training.csv", low_memory=False, index_col="article_number")
test = pd.read_csv("test.csv", low_memory=False, index_col="article_number")
from imblearn.over_sampling import SMOTE


# %%
# # Create Ordinal Encoding
# le = LabelEncoder().fit(train.topic)
# train["label"] = le.transform(train.topic)
# test["label"] = le.transform(test.topic)

# Split into x and y
train_x = train.drop([ "topic"], axis=1)
test_x = test.drop([ "topic"], axis=1)
train_y = train["topic"]
test_y = test["topic"]

topics=sorted([t for t in train['topic'].unique()])

# %% [markdown]
# ## Word Representation + Additional Processing
# Here we select how the words will be converted into input for the model. E.g. bag of words, word2vec, TF-IDF etc.
# If you would like to any additional preprocessing, this is the place to do it as well.

# %%
# Additional preprocessing (none for this example)


# %%
# Create 3 representations for the documents: Bag of Words, TF, TF-IDF.
# Parameters of each repr need to be tuned separately


tfidf = TfidfVectorizer(max_features=500).fit(train_x.article_words)

tf = TfidfVectorizer(max_features=500, use_idf=False).fit(train_x.article_words)

bow = CountVectorizer(max_features=500).fit(train_x.article_words)

# Transform words and convert from sparse matrix to array
train_tfidf = csr_matrix.toarray(tfidf.transform(train_x.article_words))
test_tfidf = csr_matrix.toarray(tfidf.transform(test_x.article_words))
#train_tfidf, train_y = SMOTE().fit_resample(train_tfidf, train_y)
# train_tfidf=SelectKBest(chi2,k=2000).fit_transform(train_tfidf,train_y)
# test_tfidf=SelectKBest(chi2,k=2000).fit_transform(test_tfidf,test_y)

train_tf = csr_matrix.toarray(tf.transform(train_x.article_words))
test_tf = csr_matrix.toarray(tf.transform(test_x.article_words))


train_bow = csr_matrix.toarray(bow.transform(train_x.article_words))
test_bow = csr_matrix.toarray(bow.transform(test_x.article_words))


model_type = LogisticRegression
word_reps = {"tfidf": train_tfidf, "tf": train_tf, "bow": train_bow}


def objective(args):
    model_type = LogisticRegression
    words = word_reps[args.pop("word_rep")]
    model = model_type(**args)
    return -np.mean(cross_val_score(model, words, train_y, cv=3, scoring='f1_macro'))


# Define a search space - Logistic regression so lets search for a value of C and which penalty to use.
# For full list of configuration options see http://hyperopt.github.io/hyperopt/getting-started/search_spaces/
space = {
     
	#"solver":hp.choice("solver",["lbfgs","newton-cg"]),
	#"multi_class":hp.choice("multi_class",["multinomial","auto"]),
	"C": hp.choice("C", np.arange(0,20,0.5)),
    "penalty": hp.choice("penalty", ["l1","l2"]),
    "word_rep": hp.choice("word_rep", ["tfidf", "tf", "bow"]),
	"class_weight":hp.choice("class_weight",["balanced"]),
	"max_iter":hp.choice("max_iter",[700])
	#"random_state":hp.choice("random_state",[20])
}

words = train_tfidf
test_words = test_tfidf

model=LogisticRegression(C= 16, class_weight='balanced',penalty='l2',max_iter=700)
model.fit(words, train_y)
print(
     classification_report(test_y, model.predict(test_words))
)
