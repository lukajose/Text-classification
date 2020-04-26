# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %%
# Importing Libraries
import json

import hyperopt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from hyperopt import fmin, hp, space_eval, tpe
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    plot_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

# Change default plot size
matplotlib.rcParams["figure.figsize"] = (12, 6)

# Setting random seed
np.random.seed(42)

# %%
# Importing data
train = pd.read_csv("training.csv", low_memory=False, index_col="article_number")
test = pd.read_csv("test.csv", low_memory=False, index_col="article_number")

# %%
# Create Ordinal Encoding
le = LabelEncoder().fit(train.topic)
train["label"] = le.transform(train.topic)
test["label"] = le.transform(test.topic)

# Split into x and y
train_x = train.drop(["label", "topic"], axis=1)
test_x = test.drop(["label", "topic"], axis=1)
train_y = train["label"]
test_y = test["label"]

# %%
# Extract best hyperparameters
with open("final_parameters.json", "r") as f:
    params = json.load(f)

# %%
# Create optimised word representation
tfidf = TfidfVectorizer(**params["tfidf_params"]).fit(train_x.article_words)

# Transform words and convert from sparse matrix to array
train_words = csr_matrix.toarray(tfidf.transform(train_x.article_words))
test_words = csr_matrix.toarray(tfidf.transform(test_x.article_words))

# Calculate per sample class weights for XGBoost
class_weights = compute_class_weight("balanced", np.unique(train_y), train_y)
train_weights = class_weights[train_y]

# Initialise models with optimal params. XGBoost needs to be calibrated before it can be ensembled
xgb = XGBClassifier(scale_pos_weight=train_weights, **params["xgb_params"])
lr = LogisticRegression(**params["lr_params"]).fit(train_words, train_y)

# %%
# Explore feature importance of individual models
# Get mapping of feature index to word.
feature_to_words = dict((value, key) for key, value in tfidf.vocabulary_.items())

#%%
# XGBoost top `num_features` features
num_features = 20
plt.bar(
    np.flip(
        [
            feature_to_words[x]
            for x in np.argsort(xgb.feature_importances_)[-num_features:]
        ]
    ),
    np.flip(
        xgb.feature_importances_[np.argsort(xgb.feature_importances_)[-num_features:]]
    ),
)
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Importance Weight")
plt.title("Top 20 features for XGBoost")
plt.savefig("xgboost_feature_importance.png", dpi=600, bbox_inces="tight")
# %%
# Logistic regression gives us top features by class
plt.figure(figsize=(30, 25))
classes = le.classes_.tolist()
classes.remove("IRRELEVANT")
for topic, i in zip(classes, range(len(classes))):
    plt.subplot(4, 3, i + 1)
    plt.bar(
        np.flip([feature_to_words[x] for x in np.argsort(lr.coef_[i])[-num_features:]]),
        np.flip(lr.coef_[i][np.argsort(lr.coef_[i])[-num_features:]]),
    )
    plt.xticks(rotation=75)
    plt.ylabel("Importance Weight")
    plt.title(f"{topic}")
plt.subplots_adjust(hspace=0.3)
plt.savefig("lr_feature_importance.png", dpi=300, bbox_inces="tight")
# %%
