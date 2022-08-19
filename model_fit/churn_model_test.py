import pandas as pd
import numpy as np
import pickle 
import time
import warnings
warnings.filterwarnings("ignore")

from churn_model_fit import include_timestamps
from churn_model_fit import predictions

print("reading in test dataset, consisting of data for January-June 2021")
X_test = pd.read_csv("../data/Tabla_01_test_English.csv", index_col=0, parse_dates=True) # time period 01.06. - 31.08.2021

na_total = X_test.isna().sum().sum()
na_prop = na_total/np.product(X_test.shape)
time.sleep(2)
print("the percentage of missing data over complete dataset is {}".format(na_prop))
time.sleep(2)
print("dropping all NAs")
X_test = X_test.dropna(inplace=True)
time.sleep(2)

# TO DO: find out why .dropna() returns None after being called

print("renaming cols")
X_test.rename(columns={
    "LO_Active_Employee_Prior6Months":"LO_Still_Hired_Prior6Months",
    "LO_Active_Employee_Post3Months":"LO_Still_Hired_Post3Months",
    "LO_Active_Employee_Post6Months":"LO_Still_Hired_Post6Months"
    },
    inplace=True
)

time.sleep(2)
print("creating X and y based on feature importance")
X = X_test[["LO_Still_Hired_Post3Months", "LO_Still_Hired_Prior6Months","LO_Still_Hired_Post6Months"]] # features selected based on calculation of feature importance in NB "all features".
y = X_test["Client_Status_Post3Months"]

time.sleep(2)
print("inserting timestamps")
X_test_timestamped = include_timestamps(X_test)

time.sleep(2)
print("reading in fit model: churn-model.bin")
with open("../artefacts/churn-model.bin", "rb") as file_in:
    clf_LR = pickle.load(file_in)

time.sleep(2)
print("making predictions on X_test")
pred_X_test = predictions(
    fit_model = clf_LR,
    X = X_test,
    y = y
    )

time.sleep(2)
print("based on new dataset, the following clients will churn in the next 3 months with a probability larger than 50%")
print(pred_X_test)