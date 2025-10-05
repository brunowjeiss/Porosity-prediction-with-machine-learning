# Porosity-prediction-with-machine-learning
Porosity prediction for Brazilian Pre-Salt carbonate fields integrating well logs and acoustic image logs.

This repository contains the Python code and resources developed for the Machine Learning component of a PhD Thesis focused on predicting rock porosity using well log data. The project explores the efficacy of several machine learning models, including Random Forest, XGBoost, and Support Vector Machines (SVM), and details the feature engineering and hyperparameter tuning processes used to refine their predictive performance.

# Project Overview
The core objective is to accurately estimate porosity, a critical reservoir property, by integrating various well log measurements. The data is supplied as CSV files containing depth-indexed well log readings (e.g., Gamma Ray, Density, Neutron, etc.) and Porosity readings from sidewall core.

#Key Steps:
Data Ingestion: Reading and preparation of well log data from CSV files.
Exploratory Data Analysis: Creating cross plots, checking biases, removing outliers, verifying correlations.
Feature Engineering: Creation of new, more informative features from raw log data (e.g. log ratios).
Model Implementation: Training and testing of Random Forest, XGBoost, and SVR regression models.
Hyperparameter Optimization: Using techniques like Grid Search or Bayesian Optimization to find the optimal settings for each model.
Evaluation: Assessing model performance using appropriate regression metrics (e.g., Root Mean Square Error (RMSE), R 
2 score, Learning rate).
