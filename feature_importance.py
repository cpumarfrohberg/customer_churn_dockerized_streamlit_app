import time, pickle, logging
from warnings import filterwarnings
filterwarnings(action='ignore')

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

logging.basicConfig(level = logging.DEBUG)

from utils import DataModeler


def main():
    model_data = DataModeler()
    prepped_data = model_data.prepare_data()
    time.sleep(2)
    split_dict = model_data.split_timestamp_data(X = prepped_data['feature_matrix'], 
                                                y = prepped_data['labels'])
    logging.debug(f'Keys of split_dict are {split_dict.keys()} and vals: {split_dict.values()}')
    logging.debug(f" Vals of 'X_train' key is {split_dict['X_train']}.")

    # logging.debug(f"Vals of key 'X_train' in 'split_dict' are {split_dict.get('X_train')}")
    X_train, X_val, y_train, y_val = split_dict.get('X_train'), split_dict.get('X_val'), split_dict.get('y_train'), split_dict.get('y_val')
    
    # time.sleep(2)
    # logging.debug("Extracting features")
    
    # time.sleep(2)
    # logging.info(f"Identify feature importance based on 'mean decrease impurity'.")
    # feature_names = [f'feature {i}' for i in range(X_train.shape[1])]
    # with open('./artifacts/churn-model.bin', 'rb') as file_in:
    #     model = pickle.load(file_in)
    # importances = model.feature_importances_
    
    # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    # pd.DataFrame({'importance': model.feature_importances_, 'feature': X_train.columns}).\
    #                 sort_values('importance', ascending=False)

    # result = permutation_importance(
    # model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=2)

    # time.sleep(2)
    # logging.info(f"Identify feature importance based on 'permutation importance'.")
    
    # model_importances = pd.Series(result.importances_mean, index=feature_names)

    # importances = model.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    # pd.DataFrame({'importance': model_importances, 'feature': X_train.columns}).\
    #                 sort_values('importance', ascending=False)


if __name__ == "__main__":
    main()
    
def test_split_timestamped():
    model_data = DataModeler()
    prepped_data = model_data.prepare_data()
    split_dict = model_data.split_timestamp_data(X = prepped_data['feature_matrix'], 
                                                y = prepped_data['labels'])
    #X_train, X_val, y_train, y_val = split_dict.get('X_train'), split_dict.get('X_val'), split_dict.get('y_train'), split_dict.get('y_val')
    assert type(split_dict) == dict

    





