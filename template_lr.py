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
# For this example lets use LogisticRegression


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
        results = results.sort_values(by=category, ascending=False)
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
model_type = LogisticRegression
word_reps = {"tfidf": train_tfidf, "tf": train_tf, "bow": train_bow}


def objective(args):
    model_type = args.pop("model_type")
    words = word_reps[args.pop("word_rep")]
    model = model_type(**args)

    return -np.mean(cross_val_score(model, words, train_y, cv=3, scoring=precision10_score))


# Define a search space - Logistic regression so lets search for a value of C and which penalty to use.
# For full list of configuration options see http://hyperopt.github.io/hyperopt/getting-started/search_spaces/
space = {
     
	#"solver":hp.choice("solver",["lbfgs","newton-cg"]),
	#"multi_class":hp.choice("multi_class",["multinomial","auto"]),
	"C": hp.choice("C", np.arange(0,20,0.5)),
    #"penalty": hp.choice("penalty", ["l2", "none"]),
    "word_rep": hp.choice("word_rep", ["tfidf", "tf", "bow"]),
	"class_weight":hp.choice("class_weight",["balanced"])
	#"max_iter":hp.choice("max_iter",[5000]),
	#"random_state":hp.choice("random_state",[20])
}
# minimize the objective over the space
#best = fmin(objective, space, algo=tpe.suggest, max_evals=10)
#print(hyperopt.space_eval(space, best))

# %% [markdown]
# ## Evaluation
# Evaluate the models by the metrics given in the example piece of code.

# %%
# Create model with best hyperparameters seen above. Need to manually select which word rep to use
#{'C': 3.0, 'class_weight': 'balanced', 'model_type': <class 'sklearn.linear_model.logistic.LogisticRegression'>, 'word_rep': 'tfidf'}

words = train_tfidf
test_words = test_tfidf

#model = model_type(**args)
model=LogisticRegression(C= 3.0, class_weight='balanced')
model.fit(words, train_y)
print(
    classification_report(test_y, model.predict(test_words), target_names=le.classes_)
)


# %%
