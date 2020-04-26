import numpy as np
np.random.seed(42)

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score , classification_report

train = pd.read_csv("training.csv")
test = pd.read_csv("test.csv")

X_train, y_train, X_test, y_test = train.article_words, train.topic, test.article_words, test.topic

clf = DecisionTreeClassifier(max_features=2000, min_samples_leaf=1, min_samples_split=4)

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


'''
accuracy 0.688
                                  precision    recall  f1-score   support

      ARTS CULTURE ENTERTAINMENT       0.50      0.67      0.57         3
BIOGRAPHIES PERSONALITIES PEOPLE       0.33      0.20      0.25        15
                         DEFENCE       0.75      0.46      0.57        13
                DOMESTIC MARKETS       0.00      0.00      0.00         2
                   FOREX MARKETS       0.51      0.52      0.52        48
                          HEALTH       0.78      0.50      0.61        14
                      IRRELEVANT       0.79      0.79      0.79       266
                   MONEY MARKETS       0.47      0.52      0.50        69
          SCIENCE AND TECHNOLOGY       0.20      0.33      0.25         3
                  SHARE LISTINGS       0.25      0.29      0.27         7
                          SPORTS       0.85      0.88      0.87        60

                        accuracy                           0.69       500
                       macro avg       0.49      0.47      0.47       500
                    weighted avg       0.70      0.69      0.69       500
'''
