# -*- coding: utf-8 -*-
# Libraries for data manipulation and machine learning components
import pandas as pd
import numpy as np
import os

# --- 1. Define Paths and Load Data ---
DATA_DIR = 'data/processed'
BLIND_TEST_DIR = 'data/blind_test' # Suggested folder for blind test data
INPUT_FILENAME = 'df_consolidado_comLL69.csv'
BLIND_FILENAME = 'df_496_BLIND.csv' # Assuming the blind test file is a CSV

INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILENAME)
BLIND_PATH = os.path.join(BLIND_TEST_DIR, BLIND_FILENAME)



# Load training data
try:
    df = pd.read_csv(INPUT_PATH)
    print(f"Training data loaded from {INPUT_PATH}")
except FileNotFoundError:
    print(f"ERROR: Training data not found at {INPUT_PATH}.")
    exit()

# Load blind test data
try:
    df_blind = pd.read_csv(BLIND_PATH)
    # Set 'PROF' as index, as done later in the original code
    df_blind.set_index('PROF', inplace=True)
    print(f"Blind test data loaded from {BLIND_PATH}")
except FileNotFoundError:
    print(f"ERROR: Blind test data not found at {BLIND_PATH}.")
    # Do not exit, as the model training might still be run without blind testing

# --- 2. Data Subsetting (Optional but retained for logic) ---
# Subsetting and concatenating all loaded wells
print("Subsetting and concatenating wells...")
well_list = ['3-BRSA-496-RJS', '9-BRSA-716-RJS', '3-BRSA-755A-RJS', 
             '3-BRSA-795-RJS', '9-LL-7-RJS', '7-LL-69-RJS']

df_subset = pd.concat([df[df['Poço'] == w] for w in well_list], ignore_index=True)
df = df_subset

# --- 3. Outlier Removal (Data Cleaning) ---
print("Applying outlier removal filters...")

# FILTER 1: DIFFERENCE POR LAB AND PHIT (PHIT * 100)
df['PHIT_percent'] = df['PHIT'] * 100
df['DIF_PHIT'] = df['POR_LAB'] - df['PHIT_percent']

df_filtered_phit = df[df["DIF_PHIT"] < 5.0]
df_filtered_phit = df_filtered_phit[df_filtered_phit["DIF_PHIT"] > -5.0]

df_filtered_phit = df_filtered_phit.drop(columns=['PHIT_percent', 'DIF_PHIT'], axis=1)
df = df_filtered_phit # Overwrite df

# FILTER 2: DIFFERENCE POR LAB AND VUGGY (VUGGY_INDEX * 100)
df['VUGGY_percent'] = df['VUGGY_INDEX'] * 100
df['DIF_VUGGY'] = df['POR_LAB'] - df['VUGGY_percent']

df_filtered_vuggy = df[df["DIF_VUGGY"] < 4.7]
df_filtered_vuggy = df_filtered_vuggy[df_filtered_vuggy["DIF_VUGGY"] > -4.7]

df_filtered_vuggy = df_filtered_vuggy.drop(columns=['VUGGY_percent', 'DIF_VUGGY'], axis=1)
df = df_filtered_vuggy # Final cleaned training DataFrame


# ==============================================================================
# 4. CORE FEATURE ENGINEERING (Creating Ratio Features)
# ==============================================================================
print("Performing Feature Engineering (Ratio Features)...")

# --- A. Training Data (df) ---
# Create VUGGY/RHOB
df['VUGGY/RHOB'] = df['VUGGY_INDEX'] / df['RHOB']

# Create VUGGY/GR
df['VUGGY/GR'] = df['VUGGY_INDEX'] / df['GR']

# Create PHIT/RHOB
df['PHIT/RHOB'] = df['PHIT'] / df['RHOB']

# --- B. Blind Test Data (df_blind) ---
if 'df_blind' in locals():
    # Create the EXACT SAME new features for the blind test set
    df_blind['VUGGY/RHOB'] = df_blind['VUGGY_INDEX'] / df_blind['RHOB']
    df_blind['VUGGY/GR'] = df_blind['VUGGY_INDEX'] / df_blind['GR']
    df_blind['PHIT/RHOB'] = df_blind['PHIT'] / df_blind['RHOB']
    print("Feature engineering applied to blind test data.")


# --- 5. Split into Features (X) and Target (y) ---
# Exclude identifier/metadata columns and the target columns
X = df.drop(columns=['Poço', 'PROF', 'KTIM', 'POR_LAB', 'PERM_LAB'], axis=1)
y = df['POR_LAB']

# Display X columns to confirm new features are present
print("\nFinal Feature Set for Model Training:")
print(list(X.columns))

# The code would proceed here with TRAIN TEST SPLIT, SCALING, and MODELING...