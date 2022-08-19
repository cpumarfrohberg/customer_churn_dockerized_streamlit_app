<h1 align="center">Predicting Customer Churn for a Savings Bank in Peru</h1>
<p align="center">Project Description</p>
This project is based on a cooperation with a Savings Bank ("Caja Municipal de Ahorro y Crédito") in Peru. Its main objective consists in predicting the likelihood that "Small and Medium Enterprise (SME)" clients of the Savings Bank will leave the institution 3 months into the future. 

<img src ="images/CMAC.jpg" width = "100">

* 3 models have been compared to each other:
* 1. a Logistic Regression classifier
* 2. a Decision Tree classifier and
* 3. a Random Forest classifier.

* The features used can be grouped into three categories:
* 1. socioeconomic features
* 2. variables related to the Loan Officers (LO) attending SME clients and 
* 3. variables related to the so-called "RFM"- methodology often used in practice for classifying customers into different groups. 

* Model	performance (AUC)
**	RF	0.94
**	Tree	0.93
**	LogReg	0.88 

* Model	performance (f1 - score)
** RF	0.94
** Tree	0.92
** LogReg	0.85

* Based on an analysis of feature importance, only 5 features are relevant for this classification problem, which can be grouped into time-related features ("year" and "month"), as well as in features related to the relationship between customers and their client advisors. 
* A particular challenge was posed by the strong amount of duplicates, which has lead to rather balanced classes; this is untypical for a churn problem, and indicates that the data potentially needs to be tidied more in future.

## Content of the project
* 1. data directory
* 2. EDA (in separate notebook)
* 3. trained models
* 4. artefacts folder containing pickled model

## In order to run the models
- clone repo locally
- create an environment with the contents of the requirements.txt file (if you are using conda: install pip first via "conda install pip" and then "pip install -r requirements.txt")

## Future Updates
- [ ] deploy locally using a flask app
- [ ] dockerize and deploy on AWS Elastic Beanstalk 
- [ ] run models with a dataset representative of a churn problem (imbalanced dataset)
- [ ] run Artificial Neural Network
- [ ] expand analysis by including time series of clients' transactions with Savings Bank in order to predict potential loan amounts to be requested by recurrent customers of the institution


## Author

**Carlos Pumar-Frohberg**

- [Profile](https://github.com/cpumarfrohberg)
- [Email](mailto:cpumarfrohberg@gmail.com?subject=Hi "Hi!")


## 🤝 Support

Comments, questions and/or feedback are welcome!
