import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score

import pickle
import time

import warnings
warnings.filterwarnings("ignore")



WEIGHTS = {0:0.41, 1:0.59}     
clf_LR = LogisticRegression(class_weight = WEIGHTS, random_state=42) 

time.sleep(2)
print("reading in dataset")
df = pd.read_csv("../data/Tabla_01_English_Unique_postEDA.csv", index_col=0, parse_dates=True)

print("splitting data")
time.sleep(2)
X = df[["LO_Active_Employee_Post3Months", "LO_Active_Employee_Prior6Months","LO_Active_Employee_Post6Months"]] # features selected based on calculation of feature importance in NB "all features".
y = df["Client_Status_Post3Months"]  

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y) 
X_train_timestamped = include_timestamps(X_train)
X_val_timestamped = include_timestamps(X_val)

time.sleep(2)
print("model fit on X_train and X_val")
fit_model_X_train = model_fit(
    model = clf_LR,
    X = X_train, 
    y = y_train
    )
fit_model_X_val = model_fit(
    model = clf_LR,
    X = X_val, 
    y = y_val
    )

time.sleep(2)
print("making predictions on X_train and X_val")
pred_X_train = predictions(
    fit_model = fit_model_X_train,
    X = X_train,
    y = y_train
    )
pred_X_val = predictions(
    fit_model = fit_model_X_val,
    X = X_val,

    y = y_val
    )
f1_score_train = f1_score(y_train, pred_X_train).round(2)
f1_score_val = f1_score(y_val, pred_X_val).round(2)

time.sleep(2)
print(f"The f1 - score based on training set is: {(f1_score_train).round(2)}")
time.sleep(2)
print(f"The f1 - score based on validation set is: {(f1_score_val).round(2)}")

time.sleep(2)   
if f1_score_train < f1_score_val:
    f1_score_delta = f1_score_val - f1_score_train
    print("model underfits by {} ".format(f1_score_delta))
else:
    print("model overfits by {} ".format(abs(f1_score_delta)))

time.sleep(2)   
print(f"Confusion Matrix on validation set: \n{confusion_matrix(y_val, pred_X_val)}")
print(f"Area Under Curve (validation set): {roc_auc_score(y_val, pred_X_val).round(2)}")

time.sleep(2)   
print("proceeding to refit model on complete dataset")
fit_model_X = model_fit(
    model = clf_LR,
    X = X, 
    y = y
    )

time.sleep(2)   
print("saving full model")
with open("../artefacts/churn-model.bin", "wb") as f_out:
    pickle.dump(fit_model_X, f_out) 
time.sleep(2)   
print("full model saved as churn-model.bin")



