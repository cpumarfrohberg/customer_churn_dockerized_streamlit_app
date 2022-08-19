<h1 align="center">Predicting Customer Churn for a Savings Bank in Peru</h1>
<p align="center">Project Description</p>
This project is based on a cooperation with a Savings Bank ("Caja Municipal de Ahorro y Crédito") in Peru. Its main objective consists in predicting the likelihood that "Small and Medium Enterprise (SME)" clients of the Savings Bank will leave the institution 3 months into the future. 

<img src ="images/CMAC.jpg" width = "100">

* Initial features used can be grouped into three categories:
* 1. socioeconomic features
* 2. variables related to the Loan Officers (LO) attending SME clients and 
* 3. variables related to the so-called "RFM"- methodology often used in practice for classifying customers into different groups. 

* However, and based on model fit of a Random Forest Classifier, the following features resulted as having greatest importance
* 1. what was the employment status of the LO 6 months prior?
* 2. what is the expectation of the app user regarding the employment status of the LO 3 months into the future?
* 3. what is the expectation of the app user regarding the employment status of the LO 6 months into the future?

## Content of the project
* 1. data directory
* 2. EDA (in separate notebook)
* 3. model fit (including feature importance with Random Forest Classifier)
* 4. artefacts folder containing pickled model
* 5. streamlit app

## In order to run the models
- clone repo locally
- create an environment with the contents of the requirements.txt file (if you are using conda: install pip first via "conda install pip" and then "pip install -r requirements.txt")

## Future Updates
- [ ] dockerize
- [ ] run Artificial Neural Network

## Author

**Carlos Pumar-Frohberg**

- [Profile](https://github.com/cpumarfrohberg)
- [Email](mailto:cpumarfrohberg@gmail.com?subject=Hi "Hi!")


## 🤝 Support

Comments, questions and/or feedback are welcome!
