# AI-Powered-Telecom-Customer-Churn-Prediction-System

# Customer Churn Prediction System

A complete end-to-end Machine Learning pipeline designed to predict customer churn using structured telecom data.
The project covers data preprocessing, feature engineering, model training, evaluation, and deployment through a Flask web application.



## Project Overview

Customer churn prediction is a classification problem aimed at identifying customers who are likely to discontinue a service.

## Dataset Overview

The dataset consists of 7,043 customer records with 23 features capturing demographic details, service subscriptions, billing information, and account tenure.
It includes both categorical and numerical variables such as contract type, internet service, monthly charges, total charges, SIM provider, and join year.
The target variable, Churn, indicates whether a customer discontinued the service, making the dataset suitable for supervised classification modeling.
This structured customer-level data supports comprehensive analysis of behavioral patterns influencing churn.

## This project implements:

* Data cleaning and preprocessing
* Advanced feature transformation
* Outlier handling
* Class imbalance treatment
* Model comparison across multiple algorithms
* Final model selection and persistence
* Deployment using Flask



## Project Architecture


Customer Churn Prediction
│
├── Data Preprocessing

│   ├── Missing Value Handling

│   ├── Variable Transformation

│   ├── Outlier Handling

│   ├── Encoding

│   ├── Feature Selection

│   └── Correlation Filtering
│
├── Data Balancing (SMOTE)
│
├── Model Training & Evaluation

│   ├── KNN

│   ├── Naive Bayes 

│   ├── Logistic Regression

│   ├── Decision Tree 

│   ├── Random Forest

│   ├── XGBoost

│   └── SVM
│
├── Model Selection (AUC-ROC Comparison)
│
├── Model Persistence (Pickle)
│
└── Flask Web Deployment




## Core Modules

### 1. Model Training Pipeline

Designed a structured end-to-end pipeline to streamline data preprocessing, feature engineering, model training, and evaluation.
Ensured modular implementation to support scalability, reproducibility, and efficient experimentation.
Integrated validation mechanisms to maintain consistency, reliability, and production readiness.

Implemented in:

Developed using Python with a modular architecture to ensure clarity, maintainability, and scalability.
Integrated core libraries such as scikit-learn, pandas, and NumPy for data processing and model development.
Structured the implementation to support seamless experimentation, evaluation, and deployment workflows. 

The CHURN class performs:

* Dataset loading
* Train-test split
* Missing value treatment
* Feature transformations
* Outlier capping
* Encoding and feature selection
* SMOTE balancing
* Feature scaling
* Multi-model evaluation
* Final Logistic Regression model training
* Model serialization (churn_model.pkl)



### 2. Machine Learning Algorithms

Evaluated multiple supervised classification algorithms to determine the most effective approach for churn prediction.
Compared model performance using standard evaluation metrics to ensure objective assessment.
Selected the final model based on predictive accuracy, stability, and alignment with business objectives. 

Models implemented:

* K-Nearest Neighbors
* Gaussian Naive Bayes
* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Support Vector Machine

Each model logs:

* Accuracy
* Confusion Matrix
* Classification Report

AUC-ROC curves are plotted to compare performance.



### 3. Data Balancing

Analyzed the class distribution to identify imbalance in the target variable.
Applied resampling techniques to improve minority class representation and reduce model bias.
This ensured balanced learning, enhanced recall for churn cases, and improved overall classification performance. 

SMOTE is applied to address class imbalance before model training.



### 4. Missing Value Handling

Identified and assessed incomplete records to determine the extent and impact of missing data.
Applied appropriate imputation strategies based on feature type and distribution to maintain dataset consistency.
This ensured data completeness, preserved analytical integrity, and supported reliable model training outcomes. 

Supported imputation strategies:

* Mean
* Median
* Mode
* Forward Fill
* Backward Fill
* Random Sampling



### 5. Variable Transformation

Applied transformation techniques to standardize feature scales and improve distribution characteristics.
Utilized normalization, standardization, and logarithmic adjustments where necessary to reduce skewness.
These transformations enhanced algorithm efficiency, improved convergence behavior, and strengthened overall model performance. 

Transformations applied:

* Log Transformation
* Square Root
* Standard Scaling
* Min-Max Scaling
* Power Transformation
* Quantile Transformation



### 6. Outlier Treatment

Performed statistical analysis to detect extreme observations using methods such as interquartile range and Z-score evaluation.
Applied appropriate mitigation techniques, including capping and controlled transformation, based on business and analytical relevance.
This improved distribution consistency, reduced model sensitivity to anomalies, and enhanced predictive reliability. 

Winsorization (Gaussian capping) is applied to control extreme values.



### 7. Hyperparameter Tuning

Conducted systematic hyperparameter optimization to enhance model efficiency and predictive accuracy.
Utilized structured search techniques with cross-validation to identify optimal parameter configurations.
This process strengthened generalization performance, minimized overfitting, and ensured robust model selection.


GridSearchCV is used to evaluate Logistic Regression parameters.



### 8. Logging Framework

Established a centralized logging mechanism to capture execution flow across data preprocessing, model training, and prediction stages.
Recorded informational messages, warnings, and error logs to ensure traceability and operational reliability.
This framework supports efficient troubleshooting, performance monitoring, and maintainable production deployment.
 

Custom logging system records:

* Execution flow
* Model metrics
* Errors
* Debug information



### 9. Web Application (Flask)

Designed and deployed a Flask-based web application to operationalize the trained churn prediction model.
Integrated request handling, input preprocessing, and model inference within RESTful APIs for real-time predictions.
The application architecture ensures scalability, maintainability, and smooth integration with production environments. 

Features:

* HTML-based UI
* Form-based customer input
* Real-time churn prediction
* Probability output (if supported)
* Model and scaler loading
* Feature alignment with trained model



## Installation

bash
https://github.com/Swarna6656/AI-Powered-Telecom-Customer-Churn-Prediction-System.git

Install dependencies:

bash
pip install -r requirements.txt


Required Libraries:

* pandas
* numpy
* scikit-learn
* imbalanced-learn
* xgboost
* flask
* matplotlib
* seaborn
* feature-engine



## Model Training

Update dataset path inside:

python
main.py


Then execute:

bash
python main.py


Outputs:

* churn_model.pkl
* standard_scalar.pkl
* Log files



## Running the Web Application

bash
python app.py


Access in browser:


https://ai-powered-telecom-customer-churn.onrender.com




## Evaluation Metrics

* Accuracy
* Confusion Matrix
* Classification Report
* ROC Curve
* AUC Score

Model comparison ensures selection of the most reliable classifier.



## Final Model

The final selected model:

* Logistic Regression
* Standard scaled features
* SMOTE-balanced training data
* Saved as serialized pickle object

---

## Business Impact

This system enables:

* Early identification of high-risk customers
* Proactive retention campaigns
* Reduced revenue loss
* Data-driven decision making



## Future Enhancements

* Deep Learning integration (ANN / LSTM)
* Real-time prediction API
* Docker containerization
* CI/CD integration
* Cloud deployment (AWS / Azure)



## Author

Odela swarnalatha

##support
if ypu like the project give me a Star
contact:swarna1704@gmail.com
