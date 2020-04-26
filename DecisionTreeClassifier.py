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

clf = DecisionTreeClassifier(max_features=2000, min_samples_leaf=1, min_samples_split=2)

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
accuracy 0.648
                                  precision    recall  f1-score   support

      ARTS CULTURE ENTERTAINMENT       0.20      0.33      0.25         3
BIOGRAPHIES PERSONALITIES PEOPLE       0.17      0.07      0.10        15
                         DEFENCE       0.60      0.46      0.52        13
                DOMESTIC MARKETS       0.00      0.00      0.00         2
                   FOREX MARKETS       0.44      0.29      0.35        48
                          HEALTH       0.57      0.29      0.38        14
                      IRRELEVANT       0.75      0.77      0.76       266
                   MONEY MARKETS       0.44      0.57      0.50        69
          SCIENCE AND TECHNOLOGY       0.12      0.33      0.18         3
                  SHARE LISTINGS       0.44      0.57      0.50         7
                          SPORTS       0.82      0.82      0.82        60

                        accuracy                           0.65       500
                       macro avg       0.41      0.41      0.40       500
                    weighted avg       0.65      0.65      0.64       500
'''
