import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score


def model_fit(X_train, X_test, y_train, y_test):
    WEIGHTS = {0:0.41, 1:0.59}
    clf_LR = LogisticRegression(class_weight = WEIGHTS, random_state=42) 
    clf_LR.fit(X_train, y_train)
    y_score = clf_LR.score(X_test, y_test)
    y_pred = clf_LR.predict(X_test)
    print(confusion_matrix(y_test, y_pred), roc_auc_score(y_test, y_score), f1_score(y_test, y_pred))
    return clf_LR

# def predictions(fit_model, X):
#     """
#     @params:
#         - X: (training and/or validation) dataset for model predictions.
#         - fit_model: fitted model.
#     @return:
#         - pred: predicted class.
#     """
#     clf = fit_model
#     return clf.predict(X)


