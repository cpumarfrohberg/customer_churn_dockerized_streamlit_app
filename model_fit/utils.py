import pandas as pd
import matplotlib.pyplot as plt


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

def model_fit(model, X, y):
    """
    @params:
        - X: (transformed) dataset for model fit.
        - y: labels.
        - model: predictor class to be fit on X_train.
    @return:
        - fit_model: trained model.
    """
    clf = model
    fit_model = clf.fit(X, y)
    return fit_model

def predictions(fit_model, X):
    """
    @params:
        - X: (training and/or validation) dataset for model predictions.
        - fit_model: fitted model.
    @return:
        - pred: predicted class.
    """
    clf = fit_model
    return clf.predict(X)

df = pd.read_csv("../data/Tabla_01_English_Unique_postEDA.csv", index_col=0, parse_dates=True)

