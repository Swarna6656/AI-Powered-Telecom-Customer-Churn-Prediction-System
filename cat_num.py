import pandas as pd
import sys
from sklearn.preprocessing import OrdinalEncoder
from log_code import setup_logging
logger=setup_logging("categorical2_numerical")

def cat(X_train_categorical,X_test_categorical,X_train_numerical,X_test_numerical):
    try:
        logger.info(f"Categorical Columns : {X_train_categorical.columns}")
        '''
        Binary Encoding
        '''
        binary_cols=['gender','Partner','Dependents','PhoneService','PaperlessBilling']
        for col in ['Partner','Dependents','PhoneService','PaperlessBilling']:
            X_train_categorical[col]=X_train_categorical[col].map({'No':0,'Yes':1})
            X_test_categorical[col]=X_test_categorical[col].map({'No':0,'Yes':1})
        X_train_categorical['gender']=X_train_categorical['gender'].map({'Female':0,'Male':1})
        X_test_categorical['gender']=X_test_categorical['gender'].map({'Female':0,'Male':1})
        '''
        Ordinal Encoding on remaining categorical columns
        '''
        ordinal_cols=[col for col in X_train_categorical.columns if col not in binary_cols]
        odi_obj=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
        odi_obj.fit(X_train_categorical[ordinal_cols])
        values_1=odi_obj.transform(X_train_categorical[ordinal_cols])
        values_2=odi_obj.transform(X_test_categorical[ordinal_cols])
        t1=pd.DataFrame(data=values_1,columns=ordinal_cols)
        t2=pd.DataFrame(data=values_2,columns=ordinal_cols)
        X_train_categorical.reset_index(drop=True,inplace=True)
        X_test_categorical.reset_index(drop=True,inplace=True)
        t1.reset_index(drop=True,inplace=True)
        t2.reset_index(drop=True,inplace=True)
        X_train_categorical_new=pd.concat([X_train_categorical[binary_cols],t1],axis=1)
        X_test_categorical_new=pd.concat([X_test_categorical[binary_cols],t2],axis=1)
        logger.info("After Converting Categorical Data To Numerical")
        logger.info(f"X_train_cat_data : {X_train_categorical_new.shape}")
        logger.info(f"Train Columns : {X_train_categorical_new.columns}")
        logger.info(f"Train Null Values : {X_train_categorical_new.isnull().sum()}")
        logger.info(f"X_test_cat_data : {X_test_categorical_new.shape}")
        logger.info(f"Test Columns : {X_test_categorical_new.columns}")
        logger.info(f"Test Null Values : {X_test_categorical_new.isnull().sum()}")
        '''
        Combine Numerical + Categorical Data
        '''
        X_train_numerical.reset_index(drop=True,inplace=True)
        X_test_numerical.reset_index(drop=True,inplace=True)
        X_train_categorical_new.reset_index(drop=True,inplace=True)
        X_test_categorical_new.reset_index(drop=True,inplace=True)
        final_training_data=pd.concat([X_train_numerical,X_train_categorical_new],axis=1)
        final_testing_data=pd.concat([X_test_numerical,X_test_categorical_new],axis=1)
        logger.info("======== Final Dataset ========")
        logger.info(f"Final Train Data Shape : {final_training_data.shape}")
        logger.info(f"Final Train Columns : {final_training_data.columns}")
        logger.info(f"Final Train Null Values : {final_training_data.isnull().sum()}")
        logger.info(f"Final Test Data Shape : {final_testing_data.shape}")
        logger.info(f"Final Test Columns : {final_testing_data.columns}")
        logger.info(f"Final Test Null Values : {final_testing_data.isnull().sum()}")
        return final_training_data,final_testing_data
    except Exception as e:
        er_ty,er_msg,er_line=sys.exc_info()
        logger.warning(f"Error in line no : {er_line.tb_lineno} : due to : {er_ty} and reason : {er_msg}")