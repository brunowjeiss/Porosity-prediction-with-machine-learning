
from pandas import read_csv

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import scipy.sparse as sparse
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# --- Portability Fix: Use Relative Path ---

DATA_FILE_PATH = 'data/DATAFRAMES DE POÇOS/df_consolidado_vuggyeq.csv'

# Load data
try:
    df = pd.read_csv(DATA_FILE_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at '{DATA_FILE_PATH}'. Please ensure the path is correct.")
    # Exiting script gracefully if data is missing
    exit()

df.columns

# Well names
WELL_NAME_1 = 'well_1'
WELL_NAME_2 = 'well_2'
WELL_NAME_3 = 'well_3'
WELL_NAME_4 = 'well_4'
WELL_NAME_5 = 'well_5'

Well_1 = df[df['Poço'] == WELL_NAME_1]
Well_2 = df[df['Poço'] == WELL_NAME_2]
Well_3 = df[df['Poço'] == WELL_NAME_3]
Well_4 = df[df['Poço'] == WELL_NAME_4]
Well_5 = df[df['Poço'] == WELL_NAME_5]

# Concatenating wells used for training (Well 2, Well 1, Well 4, Well 3)
test = pd.concat([Well_2, Well_1, Well_4, Well_3], ignore_index=True)

df = test


# load data and arrange into Pandas dataframe

feature_names = ['GR', 'RHOB', 'NPHI', 'DT', 'PHIT', 'VUGGY_INDEX']

# Removed the redundant 'y' definition
# y = df.drop(['GR', 'RHOB', 'NPHI', 'DT', 'PHIT', 'VUGGY_INDEX'], axis=1)

# Split into features and target
X = df.drop(['Poço', 'PROF', 'KTIM', 'POR_LAB', 'PERM_LAB'], axis=1)
y = df['POR_LAB']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Scale data, otherwise model will fail.
# Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)