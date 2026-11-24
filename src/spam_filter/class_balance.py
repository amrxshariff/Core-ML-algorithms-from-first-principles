"""
Class-balance reporting and majority-class baseline comparison
for imbalanced binary classification.

Works on label arrays alone, so it is independent of how the features
were built or which model produced the predictions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support



def class_balance(y, class_names=None):
    """Step 1 — counts and proportions per class.

    Counts matter as well as proportions: 13% of 5,000 and 13% of 100
    are the same ratio but very different amounts of evidence about
    the minority class.
    """
    y = np.asarray(y)
    labels, counts = np.unique(y, return_counts=True)
    names = [class_names[l] for l in labels] if class_names else labels

    return pd.DataFrame({"count": counts,
                         "proportion": counts / counts.sum()},
                        index=pd.Index(names, name="class"))


def majority_baseline(y_train, y_test):
    """Step 2 — the degenerate model that always predicts the majority class.

    Implemented directly rather than via DummyClassifier so the mechanism
    is visible; `check_against_sklearn` below asserts the two agree.
    """
    labels, counts = np.unique(y_train, return_counts=True)
    majority_label = labels[counts.argmax()]
    return np.full(shape=len(y_test), fill_value=majority_label, dtype=y_train.dtype)


def check_against_sklearn(X_train, y_train, X_test, y_pred_baseline):
    """Parity check: the hand-rolled baseline matches sklearn's DummyClassifier."""
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    np.testing.assert_array_equal(dummy.predict(X_test), y_pred_baseline)


def split_balance(y_train, y_test, class_names=None):
    """Step 6 — did the split preserve the class ratio?

    If these columns differ noticeably, pass stratify=y to train_test_split
    (and use StratifiedKFold rather than KFold for cross-validation).
    """
    train = class_balance(y_train, class_names)["proportion"]
    test = class_balance(y_test, class_names)["proportion"]
    out = pd.DataFrame({"train": train, "test": test})
    out["abs_diff"] = (out["train"] - out["test"]).abs()
    return out


def scores(y_true, y_pred, positive_label=1):
    """Accuracy plus positive-class precision / recall / F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[positive_label], zero_division=0
    )
    return {"accuracy":  (y_true == y_pred).mean(),
            "precision": precision[0],
            "recall":    recall[0],
            "f1":        f1[0]}


def compare(y_test, y_pred_model, y_pred_baseline, positive_label=1):
    """Steps 3 and 5 — model against baseline on four metrics."""
    return pd.DataFrame({
        "majority baseline": scores(y_test, y_pred_baseline, positive_label),
        "model":             scores(y_test, y_pred_model, positive_label),
    }).T


def report(y_full, y_train, y_test, y_pred_model,
           positive_label=1, class_names=None):
    """Runs steps 1-6 and prints the table set."""
    y_pred_baseline = majority_baseline(y_train, y_test)

    print("CLASS BALANCE (full dataset)")
    print(class_balance(y_full, class_names).to_string(
        formatters={"proportion": "{:.1%}".format}), "\n")

    print("SPLIT BALANCE")
    print(split_balance(y_train, y_test, class_names).to_string(
        float_format="{:.3f}".format), "\n")

    print("MODEL vs MAJORITY BASELINE")
    print(compare(y_test, y_pred_model, y_pred_baseline, positive_label)
          .to_string(float_format="{:.3f}".format), "\n")

    acc_gap = (y_test == y_pred_model).mean() - (y_test == y_pred_baseline).mean()
    print(f"Accuracy gained over predicting the majority class every time: {acc_gap:+.1%}\n")

    print("CONFUSION MATRIX — baseline (rows: true, cols: predicted)")
    print(confusion_matrix(y_test, y_pred_baseline), "\n")
    print("CONFUSION MATRIX — model")
    print(confusion_matrix(y_test, y_pred_model))

    return y_pred_baseline


# MODEL vs MAJORITY BASELINE
#                   accuracy  precision  recall    f1
#majority baseline     0.859      0.000   0.000 0.000
#model                 0.938      0.929   0.602 0.731 

#CONFUSION MATRIX — baseline (rows: true, cols: predicted)
#[[1197    0]
# [ 196    0]] 

#CONFUSION MATRIX — model
#[[1188    9]
# [  78  118]]

# accuracy is not a good measure here as if the model learns aboslutely nothing from
# training data, it will still output an accuracy of 85.9% by just picking the majority class
# every time. A model that does learn something only increases accuracy by 7.9 percentage
# points out of a possible 14.1 percentage points to improve over, 
# pointing to the trained model picking up appx. 55% of what it could learn.
# model correctly identifies ham with a success rate of 92.9%, but incorrectly misclassified
# 9 legitmate ham emails as spam. additionally it only picked up 60.2% of spam messages and allows 
# 78 spam messages to go straight into inbox.
# Confusion matrix reads as (baseline) 1197 rows classified as majority class, 
# didn't predict any ham as spam since it was only predicting everything as ham
# missed 196 spam classifications. accuracy = TN+TP/total, precision = TP/ (FP+TP)
# recall = TP/(TP+FN). For the model, the 1188 rows were classified as ham, with 9 ham being misclassified as
# spam, and 118 spam classified as spam, but 78 spam misclassfied as ham.
# Harmonic mean is 0.731. Harmonic means are typically dragged towards the smaller value
# and pays attention to your worst score. f1 weights recall and accuracy equally, if 
# you waned to weight a ham email being passed off as spam as more consequential,
# switch to fbeta_score with beta = 0.5 to weigh precision twiced as heavily
