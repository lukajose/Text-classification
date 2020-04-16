import pandas as pd
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score , classification_report

dataset = pd.read_csv("training.csv")
testset = pd.read_csv("test.csv")

y_train = dataset['topic']
y_test = testset['topic']

X_train = dataset['article_words']
X_test = testset['article_words']

dtc = tree.DecisionTreeClassifier(
 max_features=2000, min_samples_leaf=1,
 min_samples_split=2)

pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", dtc),
    ]
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"accuracy {accuracy_score(y_test,y_pred)}")
print(classification_report(y_test, y_pred))



##
##accuracy 0.686
##                                  precision    recall  f1-score   support
##
##      ARTS CULTURE ENTERTAINMENT       1.00      0.67      0.80         3
##BIOGRAPHIES PERSONALITIES PEOPLE       0.33      0.20      0.25        15
##                         DEFENCE       0.69      0.69      0.69        13
##                DOMESTIC MARKETS       0.33      0.50      0.40         2
##                   FOREX MARKETS       0.40      0.38      0.39        48
##                          HEALTH       0.40      0.43      0.41        14
##                      IRRELEVANT       0.83      0.79      0.81       266
##                   MONEY MARKETS       0.46      0.57      0.51        69
##          SCIENCE AND TECHNOLOGY       0.00      0.00      0.00         3
##                  SHARE LISTINGS       0.44      0.57      0.50         7
##                          SPORTS       0.78      0.85      0.82        60
##
##                        accuracy                           0.69       500
##                       macro avg       0.52      0.51      0.51       500
##                    weighted avg       0.69      0.69      0.69       500
