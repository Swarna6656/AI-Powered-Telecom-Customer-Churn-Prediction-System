import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from log_code import setup_logging
from scipy.stats import yeojohnson
from sklearn.preprocessing import QuantileTransformer

logger = setup_logging("vt_outlier")


def varibale_outlier_handling(X_train_num, X_test_num):

    try:
        logger.info(f"Before Train Column Name : {X_train_num.columns.tolist()}")
        logger.info(f"Before Test Column Name : {X_test_num.columns.tolist()}")

        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        for col in X_train_num.columns.tolist():
            if col == 'MonthlyCharges' or col == 'TotalCharges_re':
                X_train_num[col + '_qt'] = qt.fit_transform(X_train_num[[col]]).ravel()
                X_test_num[col + '_qt'] = qt.transform(X_test_num[[col]]).ravel()
                X_train_num.drop(col, axis=1, inplace=True)
                X_test_num.drop(col, axis=1, inplace=True)

            elif col == 'tenure':
                X_train_num[col + '_yeo'], lam_val = yeojohnson(X_train_num[col])
                X_test_num[col + '_yeo'] = yeojohnson(X_test_num[col], lmbda=lam_val)[0]
                X_train_num.drop(col, axis=1, inplace=True)
                X_test_num.drop(col, axis=1, inplace=True)

        for col in X_train_num.columns.tolist():
            if col == 'SeniorCitizen':
                X_train_num.drop(col, axis=1, inplace=True)
                X_test_num.drop(col, axis=1, inplace=True)
                continue

            if X_train_num[col].dtype != 'object':
                iqr = X_train_num[col].quantile(0.75) - X_train_num[col].quantile(0.25)
                lower_limit = X_train_num[col].quantile(0.25) - (1.5 * iqr)
                upper_limit = X_train_num[col].quantile(0.75) + (1.5 * iqr)

                X_train_num[col + '_trim'] = np.where(X_train_num[col] < lower_limit, lower_limit, np.where(X_train_num[col] > upper_limit, upper_limit, X_train_num[col]))
                X_test_num[col + '_trim'] = np.where(X_test_num[col] < lower_limit, lower_limit, np.where(X_test_num[col] > upper_limit, upper_limit, X_test_num[col]))
                X_train_num.drop(col, axis=1, inplace=True)
                X_test_num.drop(col, axis=1, inplace=True)

        logger.info(f"After Train Column Name : {X_train_num.columns.tolist()}")
        logger.info(f"After Test Column Name : {X_test_num.columns.tolist()}")

        return X_train_num, X_test_num

    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")
