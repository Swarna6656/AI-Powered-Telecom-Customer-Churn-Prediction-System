'''
In this file we are going to take numerical columns and select best columns using
1. constant technique
2. quasi constant technique
3. hypothesis testing
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from log_code import setup_logging
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

logger = setup_logging("fs")

constant_reg = VarianceThreshold(threshold=0.0)
quasi_reg = VarianceThreshold(threshold=0.1)


def best_columns(X_train_num, X_test_num, y_train):

    try:
        logger.info(f"Before Train data : {X_train_num.shape} : {X_train_num.columns.tolist()}")
        logger.info(f"Before Test data : {X_test_num.shape} : {X_test_num.columns.tolist()}")

        constant_reg.fit(X_train_num)

        constant_remove_cols = X_train_num.columns[~constant_reg.get_support()].tolist()

        logger.info(f"Number of columns are good in constant technique : {sum(constant_reg.get_support())}")
        logger.info(f"Number of columns are bad in constant technique : {sum(~constant_reg.get_support())}")
        logger.info(f"Columns to remove in constant technique : {constant_remove_cols}")

        X_train_num = X_train_num.drop(constant_remove_cols, axis=1)
        X_test_num = X_test_num.drop(constant_remove_cols, axis=1)

        quasi_reg.fit(X_train_num)

        quasi_remove_cols = X_train_num.columns[~quasi_reg.get_support()].tolist()

        logger.info(f"Number of columns are good in quasi constant technique : {sum(quasi_reg.get_support())}")
        logger.info(f"Number of columns are bad in quasi constant technique : {sum(~quasi_reg.get_support())}")
        logger.info(f"Columns to remove in quasi constant technique : {quasi_remove_cols}")

        X_train_num = X_train_num.drop(quasi_remove_cols, axis=1)
        X_test_num = X_test_num.drop(quasi_remove_cols, axis=1)

        p_values = []

        for col in X_train_num.columns:
            p_values.append(pearsonr(X_train_num[col], y_train)[1])

        p_values = pd.Series(p_values, index=X_train_num.columns)

        logger.info(f"P values are : {p_values}")

        remove_cols = p_values[p_values > 0.05].index.tolist()

        logger.info(f"Columns to remove using hypothesis testing : {remove_cols}")

        X_train_num = X_train_num.drop(remove_cols, axis=1)
        X_test_num = X_test_num.drop(remove_cols, axis=1)

        logger.info(f"After Train data : {X_train_num.shape} : {X_train_num.columns.tolist()}")
        logger.info(f"After Test data : {X_test_num.shape} : {X_test_num.columns.tolist()}")
        logger.info(f"Final Numerical Columns Count : {X_train_num.shape[1]}")

        return X_train_num, X_test_num

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")
        return None, None