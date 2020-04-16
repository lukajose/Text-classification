import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score , classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv("training.csv")
testset = pd.read_csv("test.csv")

y_train = dataset['topic']
y_test = testset['topic']

X_train = dataset['article_words']
X_test = testset['article_words']

clf = RandomForestClassifier(bootstrap= False, max_features= 'auto', min_samples_split= 2, n_estimators= 30)

pipeline = Pipeline(
    [
        ("vect", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("clf", clf),
    ]
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"accuracy {accuracy_score(y_test,y_pred)}")
print(classification_report(y_test, y_pred))

##
##accuracy 0.732
##                                  precision    recall  f1-score   support
##
##      ARTS CULTURE ENTERTAINMENT       1.00      0.33      0.50         3
##BIOGRAPHIES PERSONALITIES PEOPLE       0.00      0.00      0.00        15
##                         DEFENCE       1.00      0.23      0.38        13
##                DOMESTIC MARKETS       0.00      0.00      0.00         2
##                   FOREX MARKETS       0.48      0.21      0.29        48
##                          HEALTH       0.67      0.14      0.24        14
##                      IRRELEVANT       0.76      0.93      0.84       266
##                   MONEY MARKETS       0.54      0.65      0.59        69
##          SCIENCE AND TECHNOLOGY       0.00      0.00      0.00         3
##                  SHARE LISTINGS       1.00      0.14      0.25         7
##                          SPORTS       0.92      0.93      0.93        60
##
##                        accuracy                           0.73       500
##                       macro avg       0.58      0.33      0.36       500
##                    weighted avg       0.70      0.73      0.69       500
##
