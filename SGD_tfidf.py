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

sns.set()

train = pd.read_csv("training.csv")
plt.figure(figsize=(10, 3))
plt.legend()
# distribution of classes
train["topic"].value_counts().plot(kind="barh", color="green")

topics = sorted(train["topic"].value_counts().keys().values.tolist())
labels = [[label, category] for label, category in enumerate(topics)]
labels  # for reference only when encoding

from sklearn import preprocessing

"""used to convert categorical to id for predicting documents. sklearn trains different binary models based on id of target """
le = preprocessing.LabelEncoder()
le.fit(train["topic"])
classes = list(le.classes_)
list(enumerate(le.classes_))

sgd_f = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", SGDClassifier()),
    ]
)

train["target"] = le.transform(train["topic"])
# Split data
X_train, X_test_dev = train.iloc[:9000, :-1], train.iloc[9000:, :-1]
y_train, y_test_dev = train.iloc[:9000, -1], train.iloc[9000:, -1]
sgd_f.fit(X_train["article_words"], y_train)
y_pred = sgd_f.predict(X_test_dev["article_words"])
print(f"accuracy {accuracy_score(y_test_dev,y_pred)}")
print(classification_report(y_test_dev, y_pred, target_names=classes))

confusion_matrix(y_test_dev, y_pred)

test = pd.read_csv("test.csv")
y_test = le.transform(test["topic"])
X_test = test["article_words"]
y_pred = sgd_f.predict(X_test)
print(f"accuracy {accuracy_score(y_test,y_pred)}")
print(classification_report(y_test, y_pred, target_names=classes))

confusion_matrix(y_test, y_pred)

sns.heatmap(confusion_matrix(y_test, y_pred))
