import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


features = pd.read_csv("./data/Tabla_01_English_Unique_postEDA.csv", sep = ",", encoding="latin1" , index_col=0, parse_dates=True)
churn_history = pd.read_csv("./data/Tabla_02_Clients_English.csv", sep = ",", encoding="latin1" , index_col=0, parse_dates=True)
rfm = df = pd.read_csv("./data/Tabla_01_English_20211020.csv", sep =",", encoding= "latin1", index_col=0, parse_dates=True) 

rfm["Year"] = rfm.index.year
rfm["Month"] = rfm.index.month


# clf = LogisticRegression()

st.title("Customer Churn Predictor")

nav = st.sidebar.radio(
    "Please chose one of the following:",
    ["Home", "EDA", "Prediction", "Contribute"])

if nav == "Home":
    st.markdown(
    """ ## Welcome to the Customer Churn Predictor page.
    ##### This project was implemented jointly with a Savings Bank in Peru.
    ##### Its scope consisted in identifying the most relevant features in predicting churn of SME clients as well as making predictions three months into the future.
    """
    )
    st.image("CMAC.jpg", width=100)

if nav == "EDA":
    st.write("Welcome to the section on Exploratory Data Analysis.")
    if st.checkbox("Click here to see the time series of customer churn"):
        val = st.slider("Filter data using years", 2018, 2021)
        p = churn_history["Client_Churn"]
        churn_data = p.loc[churn_history.index.year >= val]
        st.line_chart(churn_data)
    if st.checkbox("Click here to see the time series of monetary features"):
        val = st.slider("Filter data using years", 2018, 2021)
        p2 = rfm.groupby(["Year","Month"])["n(Loans)_Outstanding_Maynas"].mean()
        churn_data = p2.loc[churn_history.index.year >= val]
        st.line_chart(churn_data)
    # if st.checkbox("Click here to see the distribution of socioeconomic features"):
    #     val = st.slider("Filter data using years", 2018, 2021)
    #     p3 = churn_history["Client_Churn"]
    #     churn_data = p3.loc[churn_history.index.year >= val]
    #     st.line_chart(churn_data)


if nav == "Prediction":
    st.markdown(
    """ #### Welcome to the predictions page.
    """
    )
    if st.checkbox("If you are interested to see features used for training the model, click here:"):
        st.table(features)
    val = st.number_input("Enter the amount of months that the Loan Officer has left your institution",
    0,
    60,
    step = 1)
    # pred = clf.predict(X_test)[0]

    # if st.button("Predict"):
    #     st.success(f“The predicted probability of churn is {round(pred)}:)

if nav == "Contribute":
    st.header("Thank you for contributing to our dataset.")
    departure_LO = st.number_input("Enter the number of months that your LO has left the bank", 0, 24)
    departure_client = st.number_input("Enter the number of months that your client has left the bank", 0, 24)
    if st.button("Submit"):
        to_add = {
            "TimeDepartureLO": departure_LO, 
            "TimeDepartureClient": departure_client,
            }
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("./data/Tabla_01_English_Unique_postEDA.csv", mode = "a", header = False, index = False)
        st.success("Submitted")


