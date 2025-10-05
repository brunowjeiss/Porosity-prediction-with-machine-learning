# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:37:53 2020

@author: bruno
"""

# Import libraries
import lasio
import pandas as pd
import os 

# --- 1. Define Paths ---
# Use os.path.join for cross-platform compatibility
DATA_DIR = "data"
INPUT_FILENAME = "well_logs.las"  # Renamed to reflect LAS format
INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILENAME)

MODELS_DIR = "models/final"
OUTPUT_FILENAME = "processed_well1_data.csv"
OUTPUT_PATH = os.path.join(MODELS_DIR, OUTPUT_FILENAME)

# --- 2. Data Ingestion ---
print(f"Reading LAS file from: {INPUT_PATH}")

try:
    las_well1 = lasio.read(INPUT_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {INPUT_PATH}. Check your directory structure.")
    

# Convert .LAS to DATAFRAME
df_well1 = las_well1.df()

# --- 3. Data Cleaning and Processing ---
# Verify if there are null values (optional: for inspection)
# print("Null counts before cleaning:\n", df_well1.isnull().sum()) 

# Keep only relevant columns and drop null values on key target/feature columns
# Assuming 'POR_LAB' and 'PERM_LAB' are the target variables for ML
df_well1_clear = df_well1.dropna(subset=['POR_LAB', 'PERM_LAB'])

# Final dropna() should be carefully considered, or use imputation
well1_data = df_well1_clear.dropna()

# --- 4. Export as .CSV ---
# Ensure the output directory exists before writing
os.makedirs(MODELS_DIR, exist_ok=True) 

print(f"Exporting processed data to: {OUTPUT_PATH}")
well1_data.to_csv(OUTPUT_PATH, index=False)
print("Processing complete.")



