##This file shows the HPO method of how we came to the final Logistic Regression model parameters
## To see the final model go to logistic_regression_final.py
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

# Setting random seed
np.random.seed(42)

# Importing data
train = pd.read_csv("training.csv", low_memory=False, index_col="article_number")
test = pd.read_csv("test.csv", low_memory=False, index_col="article_number")

# Create Ordinal Encoding
le = LabelEncoder().fit(train.topic)
train["label"] = le.transform(train.topic)
test["label"] = le.transform(test.topic)

# Split into x and y
train_x = train.drop(["label", "topic"], axis=1)
test_x = test.drop(["label", "topic"], axis=1)
train_y = train["label"]
test_y = test["label"]

# Here we select how the words will be converted into input for the model. E.g. bag of words, word2vec, TF-IDF etc.
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

#Hyper Parameter Optimisation
#experiment with word representations + hyper parameters

word_reps = {"tfidf": train_tfidf, "tf": train_tf, "bow": train_bow}

#function we wish to minimise
def objective(args):
    words = word_reps[args.pop("word_rep")]
    model = LogisticRegression(**args)
    return -np.mean(cross_val_score(model, words, train_y, cv=3, scoring='f1_macro'))

# Define a search space -
# For full list of configuration options see http://hyperopt.github.io/hyperopt/getting-started/search_spaces/
space = {    
	#"solver":hp.choice("solver",["lbfgs","newton-cg"]),
	#"multi_class":hp.choice("multi_class",["multinomial","auto"]),
	"C": hp.choice("C", np.arange(0,20,0.5)),
	"penalty": hp.choice("penalty", ["l1","l2"]),
	"word_rep": hp.choice("word_rep", ["tfidf", "tf", "bow"]),
	"class_weight":hp.choice("class_weight",["balanced"]),
	#"max_iter":hp.choice("max_iter",[700])
	#"random_state":hp.choice("random_state",[20])
}

# minimize the objective over the space
best = fmin(objective, space, algo=tpe.suggest, max_evals=10)
#Get the parameters of the best model
print(hyperopt.space_eval(space, best))

