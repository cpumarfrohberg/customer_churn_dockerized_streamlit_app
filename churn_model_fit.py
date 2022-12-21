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

