import numpy as np
np.random.seed(6)

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score , classification_report
from sklearn.ensemble import RandomForestClassifier

train  = pd.read_csv("training.csv")
test = pd.read_csv("test.csv")

X_train, y_train, X_test, y_test = train.article_words, train.topic, test.article_words, test.topic

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

'''
accuracy 0.74
                                 precision    recall  f1-score   support

      ARTS CULTURE ENTERTAINMENT       1.00      0.33      0.50         3
BIOGRAPHIES PERSONALITIES PEOPLE       0.00      0.00      0.00        15
                         DEFENCE       1.00      0.46      0.63        13
                DOMESTIC MARKETS       0.00      0.00      0.00         2
                   FOREX MARKETS       0.62      0.31      0.42        48
                          HEALTH       0.75      0.21      0.33        14
                      IRRELEVANT       0.76      0.93      0.84       266
                   MONEY MARKETS       0.56      0.65      0.60        69
          SCIENCE AND TECHNOLOGY       0.00      0.00      0.00         3
                  SHARE LISTINGS       0.00      0.00      0.00         7
                          SPORTS       0.90      0.88      0.89        60

                        accuracy                           0.74       500
                       macro avg       0.51      0.34      0.38       500
                    weighted avg       0.70      0.74      0.70       500
                   
'''
