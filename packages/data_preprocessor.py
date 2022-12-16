import pandas as pd
from sklearn.model_selection import train_test_split
import pytest

#PATH = "../data/Tabla_01_English_Unique_postEDA.csv"

def prepare_data(path_to_data):
    """
    @params:
        - path_to_data: path to data
    @return:
        - feature matrix and labels
    """
    df = pd.read_csv(path_to_data, index_col=0, parse_dates=True)
    X = df[["LO_Active_Employee_Post3Months", "LO_Active_Employee_Prior6Months","LO_Active_Employee_Post6Months"]] # features selected based on calculation of feature importance.
    y = df["Client_Status_Post3Months"]
    return {"feature_matrix": X, "labels": y}

def include_timestamps(df):
    """
    @params:
        - df: an initial DataFrame.
    @return:
        - df: a DataFrame including timestamps for months and years.
    """
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    return df

def split_data(X, y, test_size, random_state, stratify):
    """
    @params:
        - X: feature matrix
        - y: labels
        - test_size: size of test data
        - random_state: seed
        - stratify: enforce replication of initial label distribution after split 
    @return:
        - feature engineered X_train and feature engineered X_test as well as labels
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify = stratify) 
    X_train_timestamped = include_timestamps(X_train)
    X_val_timestamped = include_timestamps(X_val)
    return {
        "feature_engineered_training_set" : X_train_timestamped, 
        "feature_engineered_validation_set" : X_val_timestamped,
        "labels_training_set": y_train,
        "labels_validation_set": y_val
        }

def test_prepare_data():
    assert type(prepare_data(PATH)) == dict

def test_dataframe_shape():
    assert len(prepare_data(PATH)) == 2

prepped_data = prepare_data(PATH)
def test_split_data():
    assert type(split_data(prepped_data["feature_matrix"],
        prepped_data["labels"],
        0.2,
        42,
        prepped_data["labels"])) == dict