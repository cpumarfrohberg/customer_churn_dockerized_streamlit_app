# churn_model_fitter.py

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

