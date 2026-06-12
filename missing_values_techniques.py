import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import logging
from log_code import setup_logging
logger=setup_logging('missing_data_handling')

def random_sample_imputation_technique(X_train,X_test):
    try:
        logger.info(f'Total rows in training data : {X_train.shape}')
        logger.info(f'Total rows in testing data : {X_test.shape}')
        logger.info(f"Before technique X_train : {X_train.columns}")
        logger.info(f"Before technique X_test : {X_test.columns}")
        logger.info(f"Before technique X_train : {X_train.isnull().sum()}")
        logger.info(f"Before technique X_test : {X_test.isnull().sum()}")

        for i in X_train.columns:
            if X_train[i].isnull().sum() > 0:
                X_train[i + "_re"] = X_train[i].copy()
                X_test[i + "_re"] = X_test[i].copy()
                t1 = X_train[i].dropna().sample(X_train[i].isnull().sum(), random_state=42)
                t2 = X_test[i].dropna().sample(X_test[i].isnull().sum(), random_state=42)
                t1.index = X_train[X_train[i].isnull()].index
                t2.index = X_test[X_test[i].isnull()].index
                X_train.loc[X_train[i].isnull(), i + "_re"] = t1
                X_test.loc[X_test[i].isnull(), i + "_re"] = t2
                X_train = X_train.drop([i], axis=1)
                X_test = X_test.drop([i], axis=1)


        logger.info(f"After technique X_train : {X_train.columns}")
        logger.info(f"After technique X_test : {X_test.columns}")
        logger.info(f"After technique X_train : {X_train.isnull().sum()}")
        logger.info(f"After technique X_test : {X_test.isnull().sum()}")
        logger.info(f'Total rows in training data : {X_train.shape}')
        logger.info(f'Total rows in testing data : {X_test.shape}')

        return X_train, X_test

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no:{error_line.tb_lineno}:due to {error_msg}')