'''
In this file we are going to load the data and pass the data to all ML pipeline
to built complete Machine learning system
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings

warnings.filterwarnings("ignore")
import logging
import sys
from sklearn.model_selection import train_test_split
from missing_values_techniques import random_sample_imputation_technique
from variable_outliers import varibale_outlier_handling
from fs import best_columns
from sklearn.preprocessing import OneHotEncoder
from cat_num import cat
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from all_models import common
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from log_code import setup_logging

logger = setup_logging("main")


class CHURN_PREDICTION:
    def __init__(self, path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)  # reading the dataset
            self.df = self.df.drop(columns=['customerID'], errors='ignore')
            logger.info(f"The shape of the data : {self.df.shape}")
            logger.info(f'Data loaded successfully')
            logger.info(f'Total Rows in the data : {self.df.shape[0]}')
            logger.info(f'Total columns in the data : {self.df.shape[1]}')
            logger.info(f'Before : {self.df.isnull().sum()}')

            self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})

            # Split features and target
            self.X = self.df.iloc[:, :-2].join(self.df.iloc[:, -1])
            self.y = self.df.iloc[:, -2]

            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42)
            logger.info(f'{self.X_train.columns}')
            logger.info(f'{self.X_test.columns}')
            logger.info(f'{self.y_train.sample(5)}')
            logger.info(f'{self.y_test.sample(5)}')
            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

    def handling_missing_values(self):
        try:
            logger.info(f'Total rows in training data : {self.X_train.shape}')
            logger.info(f"Before Missing values in Train data : {self.X_train.isnull().sum()}")
            logger.info(f"Before Missing values in Test data : {self.X_test.isnull().sum()}")
            self.X_train, self.X_test = random_sample_imputation_technique(self.X_train, self.X_test)
            logger.info(f"After Missing values in Train data : {self.X_train.isnull().sum()}")
            logger.info(f"After Missing values in Test data : {self.X_test.isnull().sum()}")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in missing_values_techniques at line {error_line.tb_lineno}: {error_msg}")

    def data_separation(self):
        try:
            logger.info(f"Total data in X_train : {self.X_train.shape}")
            logger.info(f"Total data in X_test : {self.X_test.shape}")
            self.X_train_numerical = self.X_train.select_dtypes(exclude='str')
            self.X_train_categorical = self.X_train.select_dtypes(include='str')
            self.X_test_numerical = self.X_test.select_dtypes(exclude='str')
            self.X_test_categorical = self.X_test.select_dtypes(include='str')
            logger.info("============X_train_seperation_details===================")
            logger.info(f"X_train : {self.X_train.shape}")
            logger.info(f"X_train_numerical : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(
                f"X_train_categorical : {self.X_train_categorical.shape} : \n : {self.X_train_categorical.columns}")
            logger.info(f"==========X_test_seperation_details=======================")
            logger.info(f"X_test : {self.X_test.shape}")
            logger.info(f"X_test_numerical : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")
            logger.info(
                f"X_train_categorical : {self.X_test_categorical.shape} : \n : {self.X_test_categorical.columns}")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in data_separation at line {error_line.tb_lineno}: {error_msg}")

    def vt_outlier_handling(self):
        try:
            logger.info(
                f"Before X_train_numerical : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(
                f"Before X_test_numerical : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")
            self.X_train_numerical, self.X_test_numerical = varibale_outlier_handling(self.X_train_numerical,
                                                                                      self.X_test_numerical)
            logger.info(
                f"After X_train_numerical : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(
                f"After X_test_numerical : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

    def feature_selection(self):
        try:
            logger.info(
                f"Before Feature Selection : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(
                f"Before Feature Selection : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")

            self.X_train_numerical, self.X_test_numerical = best_columns(self.X_train_numerical, self.X_test_numerical,
                                                                         self.y_train)

            logger.info(
                f"After Feature Selection : {self.X_train_numerical.shape} : \n : {self.X_train_numerical.columns}")
            logger.info(
                f"After Feature Selection : {self.X_test_numerical.shape} : \n : {self.X_test_numerical.columns}")
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

    def cat_to_num(self):
        try:
            logger.info(f"Categorical columns: {self.X_train_categorical.columns}")
            self.training_data, self.testing_data = cat(self.X_train_categorical, self.X_test_categorical,
                                                        self.X_train_numerical, self.X_test_numerical)
            logger.info(self.training_data.columns.tolist())
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

    def balance_data(self):
        try:
            logger.info(f"=====Before Checking Data is Balanced or Not========")
            logger.info(f"Number of rows for Churn Customers : {1} -> : {sum(self.y_train == 1)}")
            logger.info(f"Number of rows for Not Churn Customers : {0} -> : {sum(self.y_train == 0)}")
            smote_reg = SMOTE(random_state=42)
            self.final_training_data_up, self.y_train_up = smote_reg.fit_resample(self.training_data, self.y_train)
            self.final_testing_data = self.testing_data
            logger.info(f"=====After Checking Data is Balanced or Not========")
            logger.info(f"Number of rows for Churn Customers : {1} -> : {sum(self.y_train_up == 1)}")
            logger.info(f"Number of rows for Not Churn Customers : {0} -> : {sum(self.y_train_up == 0)}")
        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")

    def scaling_data(self):
        try:
            logger.info(self.final_training_data_up)
            sc = StandardScaler()
            sc.fit(self.final_training_data_up)
            self.final_training_data_up_scaled = sc.transform(self.final_training_data_up)
            self.final_testing_data_scaled = sc.transform(self.final_testing_data)

            logger.info("===Training Logistic Regression=====")
            self.reg = LogisticRegression()
            self.reg.fit(self.final_training_data_up_scaled, self.y_train_up)
            self.y_test_predictions = self.reg.predict(self.final_testing_data_scaled)

            logger.info(f"Test data Accuracy : {accuracy_score(self.y_test, self.y_test_predictions)}")
            logger.info(f"Confusion Matrix : {confusion_matrix(self.y_test, self.y_test_predictions)}")
            logger.info(f"classification report : {classification_report(self.y_test, self.y_test_predictions)}")



            with open("standard_scaler.pkl", "wb") as f:
                pickle.dump(sc, f)

            with open("Telco_Churn_LR.pkl", "wb") as f1:
                pickle.dump(self.reg, f1)


        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")


if __name__ == "__main__":
    try:
        obj = CHURN_PREDICTION("Telco_Churn_Updated.csv")
        obj.handling_missing_values()
        obj.data_separation()
        obj.vt_outlier_handling()
        obj.feature_selection()
        obj.cat_to_num()
        obj.balance_data()
        obj.scaling_data()
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.warning(f"Error in main block line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")