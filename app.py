import pickle
import pandas as pd
import numpy as np

import streamlit as st


MODEL_FILE = "./artifacts/churn-model.bin"
INDEX = [0]


features = pd.read_csv("./data/Tabla_01_English_Unique_postEDA.csv", encoding="latin1" , index_col=0, parse_dates=True)
churn_history = pd.read_csv("./data/Tabla_02_Clients_English.csv", encoding="latin1" , index_col=0, parse_dates=True)
rfm = df = pd.read_csv("./data/Tabla_01_English_20211020.csv", encoding= "latin1", index_col=0, parse_dates=True) 

rfm["Year"] = rfm.index.year
rfm["Month"] = rfm.index.month

st.title("Customer Churn Predictor")

nav = st.sidebar.radio(
    "Please chose one of the following:",
    ["Home", "EDA", "Prediction"]
    ) 

if nav == "Home":
    st.markdown(
    """ ## Welcome to the Customer Churn Predictor page.
    ##### This project was implemented jointly with a Savings Bank in Peru.
    ##### Its main objective consists in identifying the most relevant features in predicting churn of SME clients as well as making predictions three months into the future.
    """
    )
    st.image("./images/CMAC.jpg", width=100)

if nav == "EDA":
    st.write("Welcome to the section on Exploratory Data Analysis.")
    if st.checkbox("Click here to see the time series of customer churn"):
        val = st.slider(
            "Filter data using years", 
            min_value = 2018, 
            max_value = 2021
            )
        churn_series = churn_history["Client_Churn"]
        churn_series = churn_series[churn_history.index.year >= val]
        st.line_chart(churn_series)

if nav == "Prediction":
    st.markdown(
    """ #### Welcome to the predictions page.
    """
    )
    
    if st.checkbox("<- If you are interested to see the initial data made available by the Savings Bank, click here"):
        st.table(features)
    
    st.markdown(
    """ ##### For predictions regarding your client's status in 3 months' time, please answer the following five questions. Select '0' for 'no' and '1' for 'yes' for the first three questions, and choose appropriate values for month and year.
    """
    )

    @st.cache
    def load_model():
        print("loading the model")
        with open(MODEL_FILE, "rb") as file_in:
            clf_RF = pickle.load(file_in)
        return clf_RF
    
    clf_RF = load_model()

    col1, col2, col3 = st.columns(3)

    active_past_6 = col1.number_input(
        "Was the Loan Officer employed 6 months ago?", 
        min_value = 0,
        max_value = 1,
        )
    active_post_3 = col2.number_input(
        "Do you expect the Loan Officer to still be employed in 3 months time?", 
        min_value = 0,
        max_value = 1,
        )
    active_post_6 = col3.number_input(
        "Do you expect the Loan Officer to still be employed in 6 months time?", 
        min_value = 0,
        max_value = 1,
        )

    col4, col5= st.columns(2)
    
    month = col4.number_input(
    "For which month are you making this prediction? (1=January, 2=February, etc.)", 
    min_value = 1,
    max_value = 12,
    )
    year = col5.number_input(
    "For which year are you making this prediction? (2025 onwards)", 
    min_value = 2025,
    max_value = 2030,
    )

    user_input = {
        "LO_Active_Employee_Post3Months": int(active_post_3),
        "LO_Active_Employee_Prior6Months": int(active_past_6),
        "LO_Active_Employee_Post6Months": int(active_post_6),
        "Year": int(year),
        "Month": int(month),
        }

    user_input = pd.DataFrame(
        user_input, 
        columns = user_input.keys(),
        index = INDEX
        )

    user_input = np.array(user_input).reshape(1, -1)

    pred = clf_RF.predict(user_input)[0]
    proba = clf_RF.predict_proba(user_input)[0]

    if st.button("Predict"):
        # Calculate prediction date (3 months from input date)
        from datetime import datetime, timedelta
        input_date = datetime(year, month, 1)  # First day of the month
        prediction_date = input_date + timedelta(days=90)  # Approximately 3 months
        prediction_month_year = prediction_date.strftime("%B%y")  # Format like "April26"
        
        st.success(f'Your client has a {round(proba[1], 2)} probability of leaving your institution in {prediction_month_year}.')

