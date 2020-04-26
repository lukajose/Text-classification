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
from catboost import CatBoostClassifier
from hyperopt import fmin, hp, space_eval, tpe
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

# %% [markdown]
# ## Word Representation + Additional Processing
# Here we select how the words will be converted into input for the model. E.g. bag of words, word2vec, TF-IDF etc.
# If you would like to any additional preprocessing, this is the place to do it as well.

# %%
# Create 3 representations for the documents: Bag of Words, TF, TF-IDF.
# Parameters start with simple out of box params (500 features) for simple HPO

tfidf = TfidfVectorizer(max_features=500).fit(train_x.article_words)
tf = TfidfVectorizer(max_features=500, use_idf=False).fit(train_x.article_words)
bow = CountVectorizer(max_features=500).fit(train_x.article_words)

# Transform words and convert from sparse matrix to array
train_tfidf = csr_matrix.toarray(tfidf.transform(train_x.article_words))
test_tfidf = csr_matrix.toarray(tfidf.transform(test_x.article_words))

train_tf = csr_matrix.toarray(tf.transform(train_x.article_words))
test_tf = csr_matrix.toarray(tf.transform(test_x.article_words))

train_bow = csr_matrix.toarray(bow.transform(train_x.article_words))
test_bow = csr_matrix.toarray(bow.transform(test_x.article_words))


# %% [markdown]
# ## Simple Hyper Parameter Optimisation
# This is done using the [hyperopt](https://github.com/hyperopt/hyperopt) package. The algorithm searches the hyperparameter space by minimising the f1 score across many trials. The larger the `max_evals` parameter, the higher the likelihood of obtaining the optimal hyperparameters (should be >1000 ideally). Note: HPO can take a significant amount of time.

# %%
word_reps = {"tfidf": train_tfidf, "tf": train_tf, "bow": train_bow}
class_weights = compute_class_weight("balanced", np.unique(train_y), train_y)
train_weights = class_weights[train_y]


def objective(args):
    model_type = args.pop("model_type")
    words = word_reps[args.pop("word_rep")]
    model = model_type(**args)

    return -np.mean(cross_val_score(model, words, train_y, cv=3, scoring="f1_macro"))


space = hp.choice(
    "classifier_type",
    [
        {
            "model_type": AdaBoostClassifier,
            "base_estimator": DecisionTreeClassifier(
                max_depth=1, class_weight="balanced"
            ),
            "n_estimators": hp.choice(
                "ada_n_estimators", np.arange(1, 500, 10, dtype=int)
            ),
            "learning_rate": hp.uniform("ada_learning_rate", 0, 1),
            "random_state": 42,
            "word_rep": hp.choice("ada_word_rep", ["tfidf", "tf", "bow"]),
        },
        {
            "model_type": GradientBoostingClassifier,
            "learning_rate": hp.uniform("gb_learning_rate", 0, 1),
            "max_depth": (hp.choice("gb_max_depth", np.arange(1, 10, dtype=int))),
            "random_state": 42,
            "word_rep": hp.choice("gb_word_rep", ["tfidf", "tf", "bow"]),
        },
        {
            "model_type": XGBClassifier,
            "objective": "multi:softprob",
            "n_estimators": hp.choice("n_estimators", np.arange(1, 500, 10, dtype=int)),
            "max_depth": hp.choice("max_depth", np.arange(1, 10, dtype=int)),
            "verbosity": 0,
            "random_state": 42,
            "word_rep": hp.choice("xgb_word_rep", ["tfidf", "tf", "bow"]),
        },
        {
            "model_type": CatBoostClassifier,
            "loss_function": "MultiClass",
            "n_estimators": hp.choice(
                "cat_n_estimators", np.arange(1, 500, 10, dtype=int)
            ),
            "max_depth": hp.choice("cat_max_depth", np.arange(1, 10, dtype=int)),
            "verbose": False,
            "random_state": 42,
            "word_rep": hp.choice("cat_word_rep", ["tfidf", "tf", "bow"]),
        },
    ],
)


# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=5000)
print(hyperopt.space_eval(space, best))

# Optionally save hyperparams
# pd.DataFrame(best, index=[0]).to_csv("rough_hpo.csv", index=False)

# %% [markdown]
# ## In Depth Hyper Parameter Optimisation
# Now we have selected the boosting algorithm and document representation (XGBoost and tf-idf), we can perform a more in depth HPO over both parameters

#%%
class_weights = compute_class_weight("balanced", np.unique(train_y), train_y)
train_weights = class_weights[train_y]


def objective(args):
    tfidf = TfidfVectorizer(**args["tfidf_params"])
    try:
        tfidf = tfidf.fit(train_x.article_words)
        words = tfidf.transform(train_x.article_words)
    except ValueError:
        return 0
    model = XGBClassifier(**args["model_params"])

    return -np.mean(cross_val_score(model, words, train_y, cv=5, scoring="f1_macro"))


space = {
    "model_params": {
        "objective": "multi:softprob",
        "n_estimators": hp.choice("n_estimators", np.arange(1, 500, dtype=int)),
        "max_depth": hp.choice("max_depth", np.arange(1, 20, dtype=int)),
        "learning_rate": hp.loguniform("learning_rate", 0, 1),
        "verbosity": 0,
        "gamma": hp.loguniform("gamma", 0, 2),
        "reg_alpha": hp.loguniform("reg_alpha", 0, 1),
        "reg_lambda": hp.loguniform("reg_lambda", 0, 1),
        "subsample": hp.uniform("subsample", 0, 1),
        "random_state": 42,
        "scale_pos_weight": train_weights,
        "colsample_bytree": hp.uniform("colsample_bytree", 0, 1),
        "colsample_bylevel": hp.uniform("colsample_bylevel", 0, 1),
        "colsample_bynode": hp.uniform("colsample_bynode", 0, 1),
        "min_child_weight": hp.choice("min_child_weight", np.arange(1, 50,)),
    },
    "tfidf_params": {
        "max_df": hp.uniform("max_df", 0.5, 1),
        "min_df": hp.uniform("min_df", 0, 0.5),
        # "ngram_range": (1, hp.choice("ngram_range", np.arange(1,5))),
        "max_features": 500,
    },
}

# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=5000)
print(hyperopt.space_eval(space, best))

# Optionally save best hyperparamers
# pd.DataFrame(best, index=[0]).to_csv("hp/xgb_f1macro_thorough.csv", index=False)

# %% [markdown]
# ## Evaluation
# Evaluate the models by the metrics given in the example piece of code.

# %%
# Load up best hyperparameters
args = hyperopt.space_eval(space, best)

# %%
# Initialise optimal model
model = XGBClassifier(**args["model_params"])
tfidf = TfidfVectorizer(**args["tfidf_params"]).fit(train_x.article_words)
train_words = tfidf.transform(train_x.article_words)
test_words = tfidf.transform(test_x.article_words)

# Fit to training data
model.fit(train_words, train_y)

# Print metrics
print(
    "Test",
    classification_report(test_y, model.predict(test_words), target_names=le.classes_),
)

# Get training metrics
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

# %%
# Investigate confusion matrices
plot_confusion_matrix(
    model,
    test_words,
    test_y,
    normalize="pred",
    cmap="Blues",
    display_labels=le.classes_,
    xticks_rotation=75,
)
plot_confusion_matrix(
    model,
    test_words,
    test_y,
    normalize="true",
    cmap="Blues",
    display_labels=le.classes_,
    xticks_rotation=75,
)
plot_confusion_matrix(
    model,
    test_words,
    test_y,
    cmap="Blues",
    display_labels=le.classes_,
    xticks_rotation=75,
)
