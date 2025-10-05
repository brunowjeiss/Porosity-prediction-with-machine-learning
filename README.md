# Porosity Prediction with Machine Learning

This repository contains Python code and resources developed for a novel machine learning workflow to predict rock porosity in complex porous networks. The focus is on carbonate reservoirs from Brazil's Pre-Salt fields, supporting a PhD thesis project.

## Table of Contents

- [Project Overview](#project-overview)
- [Workflow](#workflow)
- [Getting Started](#getting-started)
- [Results & Evaluation](#results--evaluation)

## Project Overview

The main objective is to accurately estimate porosity—a key reservoir property—by integrating a variety of well log measurements. Input data is provided as CSV files containing depth-indexed well logs, acoustic image logs, and sidewall core data.

## Workflow

The project follows these major steps:

1. **Data Ingestion**: Load and prepare well log data from CSV files.
2. **Exploratory Data Analysis (EDA)**: Generate cross plots, check for bias, remove outliers, and analyze feature correlations.
3. **Feature Engineering**: Create new, more informative features from raw log data (e.g., log ratios).
4. **Model Building**: Train and test several regression models, including Random Forest, XGBoost, and SVR.
5. **Hyperparameter Optimization**: Use Grid Search or Bayesian Optimization to fine-tune model parameters.
6. **Evaluation**: Assess model performance with regression metrics such as RMSE and R² score.

## Getting Started

### Prerequisites

- Python 3.7+
- Recommended: virtualenv or conda

Install requirements:

```bash
pip install -r requirements.txt
```

### Data

Place your well log CSV files in the `data/` directory.


## Results & Evaluation

Model performance is evaluated using:

- **Root Mean Square Error (RMSE)**
- **R² Score**
- Learning curves and feature importance plots

Results will be saved in the `results/` directory.

