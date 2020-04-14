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
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

train_tf = csr_matrix.toarray(tf.transform(train_x.article_words))
test_tf = csr_matrix.toarray(tf.transform(test_x.article_words))

train_bow = csr_matrix.toarray(bow.transform(train_x.article_words))
test_bow = csr_matrix.toarray(bow.transform(test_x.article_words))

# %% [markdown]
# ## Model Selection
# Choose what type of model you would like to use. Pick some hyperparameters (these will be tuned in the next section) and ensure the model runs as expected on the data.

# %%
# Try xgboost to start with
model = XGBClassifier()
model.fit(train_bow, train_y)
model.score(test_bow, test_y)

# %% [markdown]
# ## Metrics
# Here we define a custom metric that only accounts for precision among the top 10 predicted instances. Refer to docstring for more information

# %%
def precision10(y_true: pd.Series, y_pred: np.array) -> float:
    """Calculates the precision for each class over the top 10 predicted elements.

    For each class we find the predictions with the top 10 largest probabilities. If a class
    has fewer than 10 predictions than all predictions are considered. The precision is then
    calculated for these recommendations for each class. A simple average of each class's
    precision is returned 
    
    Args:
        y_true: The true class labels for each sample
        y_pred: The predicted probabilities for each sample
    
    Returns:
        float: The average precision on the top 10 or fewer predictions
    """
    y_true = pd.DataFrame(y_true.values, columns=["labels"])
    y_pred = pd.DataFrame(y_pred)
    results = y_true.join(y_pred)

    precisions = []
    for category in results.drop("labels", axis=1).columns:
        # Sort by predicted probability for each class
        results = results.sort_values(by=category, ignore_index=True, ascending=False)
        probabilities = results.drop("labels", axis=1)

        # `recommendations` is all the elements where the given class is predicted
        # If there are more than ten, just take the 10 with highest predicted probability
        recommendations = results[probabilities[category] == probabilities.max(axis=1)]
        if (len(recommendations)) > 10:
            recommendations = recommendations.head(10)

        # Calculate the predicted score for these predictions
        precisions.append(
            precision_score(
                (recommendations.labels == category).astype(int),
                y_pred=np.ones(recommendations.labels.shape),
            )
        )

    return np.mean(precisions)


precision10_score = make_scorer(precision10, needs_proba=True)

# %% [markdown]
# ## Hyper Parameter Optimisation
# This step is optional, but will use Bayesian optimisation to find the best hyperparameters in the search space. This is done using the [hyperopt](https://github.com/hyperopt/hyperopt) package. The algorithm searches the hyperparameter space by minimising the f1 score across many trials. The larger the `max_evals` parameter, the higher the likelihood of obtaining the optimal hyperparameters (should be >1000 ideally). Note: HPO can take a significant amount of time.
#
# If you are using an sklearn model, consider the package [hyperopt-sklearn](https://github.com/hyperopt/hyperopt-sklearn) which will automatically search over the supported hyperparameters of the model. This isn't used in this example however.


# %%
word_reps = {"tfidf": train_tfidf, "tf": train_tf, "bow": train_bow}
class_weights = compute_class_weight("balanced", np.unique(train_y), train_y)
train_weights = class_weights[train_y]


def objective(args):
    model_type = args.pop("model_type")
    words = word_reps[args.pop("word_rep")]
    model = model_type(**args)

    return -np.mean(
        cross_val_score(model, words, train_y, cv=3, scoring=precision10_score)
    )


# space = hp.choice(
#     "classifier_type",
#     [
#         # {
#         #     "model_type": AdaBoostClassifier,
#         #     "base_estimator": DecisionTreeClassifier(
#         #         max_depth=1, class_weight="balanced"
#         #     ),
#         #     "n_estimators": hp.choice("ada_n_estimators", np.arange(1, 500, dtype=int)),
#         #     "learning_rate": hp.uniform("ada_learning_rate", 0, 1),
#         #     "random_state": 42,
#         #     "word_rep": hp.choice("ada_word_rep", ["tfidf", "tf", "bow"]),
#         # },
#         # {
#         #     "model_type": GradientBoostingClassifier,
#         #     "subsample": hp.uniform("gb_subsample", 0, 1),
#         #     "learning_rate": hp.uniform("gb_learning_rate", 0, 1),
#         #     "max_depth": (hp.choice("gb_max_depth", np.arange(1, 10, dtype=int))),
#         #     "random_state": 42,
#         #     "word_rep": hp.choice("gb_word_rep", ["tfidf", "tf", "bow"]),
#         # },
#         {
#             "model_type": XGBClassifier,
#             "objective": "multi:softprob",
#             "n_estimators": hp.choice("n_estimators", np.arange(1, 500, dtype=int)),
#             "max_depth": hp.choice("max_depth", np.arange(1, 10, dtype=int)),
#             "learning_rate": hp.loguniform("learning_rate", 0, 1),
#             "verbosity": 0,
#             # "tree_method": "gpu_hist",
#             "gamma": hp.loguniform("gamma", 0, 1),
#             "reg_alpha": hp.loguniform("reg_alpha", 0, 1),
#             "reg_lambda": hp.loguniform("reg_lambda", 0, 1),
#             "subsample": hp.uniform("subsample", 0, 1),
#             "random_state": 42,
#             "word_rep": hp.choice("word_rep", ["tfidf", "tf", "bow"]),
#             "scale_pos_weight": train_weights,
#             "colsample_bytree": hp.uniform("colsample_bytree", 0, 1),
#             "colsample_bylevel": hp.uniform("colsample_bytree", 0, 1),
#             "colsample_bynode": hp.uniform("colsample_bytree", 0, 1),
#         },
#         # {
#         #     "model_type": CatBoostClassifier,
#         #     "loss_function": "MultiClass",
#         #     "n_estimators": hp.choice("cat_n_estimators", np.arange(1, 500, dtype=int)),
#         #     "max_depth": hp.choice("cat_max_depth", np.arange(1, 10, dtype=int)),
#         #     "learning_rate": hp.loguniform("cat_learning_rate", 0, 1),
#         #     "verbose": False,
#         #     "l2_leaf_reg": hp.loguniform("cat_reg_lambda", 0, 10),
#         #     "subsample": hp.uniform("cat_subsample", 0, 1),
#         #     "random_state": 42,
#         #     "word_rep": hp.choice("cat_word_rep", ["tfidf", "tf", "bow"]),
#         #     "bootstrap_type": "Bernoulli",
#         # },
#     ],
# )

space = {
    "model_type": XGBClassifier,
    "objective": "multi:softprob",
    "n_estimators": hp.choice("n_estimators", np.arange(1, 500, dtype=int)),
    "max_depth": hp.choice("max_depth", np.arange(1, 20, dtype=int)),
    "learning_rate": hp.loguniform("learning_rate", 0, 1),
    "verbosity": 0,
    # "tree_method": "gpu_hist",
    "gamma": hp.loguniform("gamma", 0, 2),
    "reg_alpha": hp.loguniform("reg_alpha", 0, 1),
    "reg_lambda": hp.loguniform("reg_lambda", 0, 1),
    "subsample": hp.uniform("subsample", 0, 1),
    "random_state": 42,
    "word_rep": hp.choice("word_rep", ["tfidf", "tf", "bow"]),
    "scale_pos_weight": train_weights,
    "colsample_bytree": hp.uniform("colsample_bytree", 0, 1),
    "colsample_bylevel": hp.uniform("colsample_bylevel", 0, 1),
    "colsample_bynode": hp.uniform("colsample_bynode", 0, 1),
    "min_child_weight": hp.choice("min_child_weight", np.arange(1, 50,)),
}

# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)
print(hyperopt.space_eval(space, best))
pd.DataFrame(best, index=[0]).to_csv("xgb_pr10.csv", index=False)

# %%
# Hard code best known params so far and load if no HPO
# best = pd.read_csv("xgb_best_pr_macro").to_dict()
best = {
    "colsample_bylevel": 0.4311244308631876,
    "colsample_bynode": 0.4883649316341705,
    "colsample_bytree": 0.2188455278994769,
    "gamma": 1.7012247488947432,
    "learning_rate": 1.1136910835752074,
    "max_depth": 0,
    "min_child_weight": 0,
    "n_estimators": 271,
    "reg_alpha": 2.132610221304731,
    "reg_lambda": 1.8979000825560892,
    "subsample": 0.6462993422098632,
    "word_rep": 1,
}

# %% [markdown]
# ## Evaluation
# Evaluate the models by the metrics given in the example piece of code.

# %%
# Create model with best hyperparameters seen above. Need to manually select hyperparameters

args = hyperopt.space_eval(space, best)
model_type = args.pop("model_type")
words = word_reps[args.pop("word_rep")]
test_words = test_tf

model = model_type(**args)
model.fit(words, train_y)
print(
    classification_report(test_y, model.predict(test_words), target_names=le.classes_)
)


# %%
plot_confusion_matrix(
    model,
    test_tf,
    test_y,
    normalize="pred",
    cmap="Blues",
    display_labels=le.classes_,
    xticks_rotation=75,
)


# %%
