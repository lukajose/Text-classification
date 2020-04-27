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
matplotlib.rcParams["figure.figsize"] = (12, 8)

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
xgb_calibrated = CalibratedClassifierCV(xgb, cv=5)
lr = LogisticRegression(**params["lr_params"])
model = VotingClassifier(
    [("xgboost", xgb_calibrated), ("logistic_regression", lr)], voting="soft"
)

# %%
# Compute metrics
model.fit(train_words, train_y)
print(
    "Test",
    classification_report(test_y, model.predict(test_words), target_names=le.classes_),
)

# Get test
train_scores = cross_validate(
    model,
    train_words,
    train_y,
    cv=5,
    scoring=["precision_macro", "recall_macro", "f1_macro"],
)
print("Precision", np.mean(train_scores["test_precision_macro"]))
print("Recall", np.mean(train_scores["test_recall_macro"]))
print("f1", np.mean(train_scores["test_f1_macro"]))

#%%
# Extract recommended articles
def getArticles(y_true: pd.Series, y_pred: np.array):
    """Calculates recommended articles and key metrics for the predictions

    Args:
        y_true (pd.Series): The true class labels
        y_pred (np.array): The predicted probabilities of each class per sample

    Returns:
        dict: A dictionary where each key corresponds to a class,and each value is
            a list of: [[predicted articles], precision, recall, f1]
    """
    y_true = pd.DataFrame(y_true.values, columns=["labels"], index=y_true.index)
    y_pred = pd.DataFrame(y_pred, index=y_true.index)
    results = y_true.join(y_pred)

    final_recs = {}
    for category in results.drop("labels", axis=1).columns:
        # Sort by predicted probability for each class
        results = results.sort_values(by=category, ascending=False)
        probabilities = results.drop("labels", axis=1)

        # `recommendations` is all the elements where the given class is predicted
        # If there are more than ten, just take the 10 with highest predicted probability
        recommendations = results[probabilities[category] == probabilities.max(axis=1)]
        if (len(recommendations)) > 10:
            recommendations = recommendations.head(10)

        category_recs = []
        category_recs.append(sorted(recommendations.index.values.tolist()))

        # Calculate the predicted score for these predictions
        category_recs.append(
            precision_score(
                (results.labels == category).astype(int),
                y_pred=results.index.isin(recommendations.index).astype(int),
            )
        )
        category_recs.append(
            recall_score(
                (results.labels == category).astype(int),
                y_pred=results.index.isin(recommendations.index).astype(int),
            )
        )
        category_recs.append(
            f1_score(
                (results.labels == category).astype(int),
                y_pred=results.index.isin(recommendations.index).astype(int),
            )
        )
        final_recs[le.inverse_transform([category])[0]] = category_recs
    return final_recs


# %%
results = getArticles(test_y, model.predict_proba(test_words))
print(results)
with open("final_results.json", "w") as f:
    json.dump(results, f)
