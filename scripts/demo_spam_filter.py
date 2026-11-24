"""Demo on a synthetic dataset with ~13% positive class."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from spam_filter.data import load_sms_spam
from spam_filter.class_balance import report, check_against_sklearn

X, y = load_sms_spam()   
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

vec = TfidfVectorizer()
X_train_vec = vec.fit_transform(X_train)   # fit on train only
X_test_vec  = vec.transform(X_test)        # transform test

model = LogisticRegression(max_iter=1000).fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)


y_pred_baseline = report(y, y_train, y_test, y_pred,
                         positive_label=1, class_names={0: "ham", 1: "spam"})

check_against_sklearn(X_train, y_train, X_test, y_pred_baseline)
print("\nparity with DummyClassifier: OK")


#CLASS BALANCE (full dataset)
#       count proportion
#class                  
#ham     4827      86.6%
#spam     747      13.4% 

#SPLIT BALANCE
#       train  test  abs_diff
#class                       
#ham    0.866 0.866     0.000
#spam   0.134 0.134     0.000 

#MODEL vs MAJORITY BASELINE
#                   accuracy  precision  recall    f1
#majority baseline     0.866      0.000   0.000 0.000
#model                 0.973      1.000   0.797 0.887 

#Accuracy gained over predicting the majority class every time: +10.7%

#CONFUSION MATRIX — baseline (rows: true, cols: predicted)
#[[1207    0]
# [ 187    0]] 

#CONFUSION MATRIX — model
#[[1207    0]
# [  38  149]]

# logistic regression applied improves accuracy appx. +10.7 percentage points over
# picking majority class every time. Sampling from train and test split displays equal proportions.
# Model successfully identifies ham with 100% success rate, and unseccessfully passes spam as ham
# 20.3% of the time or picked up 79.7% of spam messages as spam. Recall score brings harmonic mean down here.
