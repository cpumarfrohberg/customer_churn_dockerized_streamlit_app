import pickle
import time

import packages.data_preprocessor as dp
import packages.model_trainer as mt 

def main():
    # 0. Path to data
    PATH = "./data/Tabla_01_English_Unique_postEDA.csv"

    # 1. Prep data
    prepped_data = dp.prepare_data(PATH)

    # 2. Split data
    X_train_FE, X_val_FE, y_train, y_val = dp.split_data(
        prepped_data["feature_matrix"],
        prepped_data["labels"],
        0.2,
        42,
        prepped_data["labels"]
        )

    # 3. Fit model
    model = mt.model_fit(X_train_FE, X_val_FE, y_train, y_val)

    # 4. Save fitted model
    time.sleep(2)   
    print("saving full model")
    with open("../artefacts/churn-model.bin", "wb") as f_out:
        pickle.dump(model, f_out) 
    time.sleep(2)   
    print("full model saved as churn-model.bin")

if __name__ == "__main__":
    main()

# TODO: run model based on this refactored version



# time.sleep(2)
# print("model fit on X_train and X_val")
# fit_model_X_train = model_fit(
#     model = clf_LR,
#     X = X_train_timestamped, 
#     y = y_train
#     )
# fit_model_X_val = model_fit(
#     model = clf_LR,
#     X = X_val_timestamped, 
#     y = y_val
#     )

# time.sleep(2)
# print("making predictions on X_train and X_val")
# pred_X_train = predictions(
#     fit_model = fit_model_X_train,
#     X = X_train
#     )
# pred_X_val = predictions(
#     fit_model = fit_model_X_val,
#     X = X_val_timestamped
#     )
# f1_score_train = f1_score(y_train, pred_X_train).round(2)
# f1_score_val = f1_score(y_val, pred_X_val).round(2)

# time.sleep(2)
# print(f"The f1 - score based on training set is: {(f1_score_train).round(2)}")
# time.sleep(2)
# print(f"The f1 - score based on validation set is: {(f1_score_val).round(2)}")

# time.sleep(2)   
# if f1_score_train < f1_score_val:
#     f1_score_delta = f1_score_val - f1_score_train
#     print("model underfits by {} ".format(f1_score_delta))
# else:
#     print("model overfits by {} ".format(abs(f1_score_delta)))

# time.sleep(2)   
# print(f"Confusion Matrix on validation set: \n{confusion_matrix(y_val, pred_X_val)}")
# print(f"Area Under Curve (validation set): {roc_auc_score(y_val, pred_X_val).round(2)}")

# time.sleep(2)   
# print("proceeding to refit model on complete dataset")
# fit_model_X = model_fit(
#     model = clf_LR,
#     X = X, 
#     y = y
#     )





