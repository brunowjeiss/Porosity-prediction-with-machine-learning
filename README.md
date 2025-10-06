# Porosity Prediction with Machine Learning

This repository contains Python code and resources developed for a novel machine learning workflow to predict rock porosity in complex porous networks. The focus is on carbonate reservoirs from Brazil's Pre-Salt fields, supporting a PhD thesis project.

## Project Overview

This project focuses on accurately predicting porosity, a critical geological property, in the challenging Brazilian Pre-Salt oil and gas reservoirs. Consistent geological and petrophysical models are vital for successful exploration and production in these complex areas.

## Machine Learning Approach and Results

Supervised machine learning was used to predict porosity by combining various well-log measurements:
Conventional Logs
Nuclear Magnetic Resonance (NMR) Logs
A vuggy index (a measure of pore space shape/type) extracted from Acoustic Borehole Image Logs.
The models' predictions were validated against routine core petrophysical analyses, which is the industry standard for ground truth.

<img width="987" height="436" alt="image" src="https://github.com/user-attachments/assets/5d2f5908-c624-430f-88bc-cc8422260933" />


Algorithm	R2 Score	Root Mean Squared Error (RMSE)
Random Forest	0.835	1.75
XGBoost	0.836	1.68
Support Vector Regression (SVR)	0.8	1.82

<img width="977" height="1053" alt="image" src="https://github.com/user-attachments/assets/8c1cf5b1-cd49-4bcf-be64-756aaad9a956" />


Key Takeaway: All three models, particularly XGBoost (with an R2 of 0.836 and RMSE of 1.68), significantly outperformed the original NMR measurement. This demonstrates that the machine learning integration provides a much more accurate porosity estimate—which is more reliable and robust than traditional methods, ultimately saving time and cost in reservoir evaluation.

## Algorithms and Performance
We tested three popular machine learning algorithms: Random Forest, XGBoost, and Support Vector Regression (SVR).

## Workflow

The project follows these major steps:

1. **Data Ingestion**: Load and prepare well log data from CSV files.
2. **Exploratory Data Analysis (EDA)**: Generate cross plots, check for bias, remove outliers, and analyze feature correlations.
3. **Feature Engineering**: Create new, more informative features from raw log data (e.g., log ratios).
4. **Model Building**: Train and test several regression models, including Random Forest, XGBoost, and SVR.
5. **Hyperparameter Optimization**: Use Grid Search or Bayesian Optimization to fine-tune model parameters.
6. **Evaluation**: Assess model performance with regression metrics such as RMSE and R² score.

<img width="902" height="1222" alt="image" src="https://github.com/user-attachments/assets/79067db5-cdde-482d-9468-a6d0572a8b3c" />






