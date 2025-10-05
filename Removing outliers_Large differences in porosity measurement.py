# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:37:53 2020

@author: bruno
"""
# Libraries for data manipulation
import pandas as pd
import numpy as np
import os

# Libraries for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Define Paths and Load Data ---

# Use relative paths for portability
# Assumes the consolidated data is in the 'data/processed' directory
DATA_DIR = 'data/processed'
INPUT_FILENAME = 'df_consolidado_comLL69.csv'
INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILENAME)

# REMOVED: os.chdir('C:/Users/User/Desktop/NOVO DOUTORADO/NOVO CODIGOS')

# Load data
try:
    df = pd.read_csv(INPUT_PATH)
    print(f"Data loaded successfully from {INPUT_PATH}")
except FileNotFoundError:
    print(f"ERROR: File not found at {INPUT_PATH}. Check your directory structure.")
    exit()


# --- 2. Outlier Removal (POR_LAB vs PHIT) ---
# Calculate the percentage version of PHIT and the difference
df['PHIT_percent'] = df['PHIT'] * 100
df['DIF'] = df['POR_LAB'] - df['PHIT_percent']

# Filter 1: Remove outliers based on PHIT difference (e.g., |POR_LAB - PHIT*100| <= 5.0)
df_filtered_phit = df[df["DIF"] < 5.0]
df_filtered_phit = df_filtered_phit[df_filtered_phit["DIF"] > -5.0]

# Clean up temporary columns and reassign DataFrame
df_filtered_phit = df_filtered_phit.drop(columns=['PHIT_percent', 'DIF'], axis=1)
df = df_filtered_phit


# --- 3. Outlier Removal (POR_LAB vs VUGGY) ---
# Calculate the percentage version of VUGGY_INDEX and the difference
df['VUGGY_percent'] = df['VUGGY_INDEX'] * 100
df['DIF_VUGGY'] = df['POR_LAB'] - df['VUGGY_percent']

# Filter 2: Remove outliers based on VUGGY_INDEX difference (e.g., |POR_LAB - VUGGY*100| <= 4.7)
df_filtered_vuggy = df[df["DIF_VUGGY"] < 4.7]
df_filtered_vuggy = df_filtered_vuggy[df_filtered_vuggy["DIF_VUGGY"] > -4.7]

# Clean up temporary columns and reassign DataFrame
df_filtered_vuggy = df_filtered_vuggy.drop(columns=['VUGGY_percent', 'DIF_VUGGY'], axis=1)
df = df_filtered_vuggy


# --- 4. Prepare Data for Plotting ---

# Keep only columns desired for plotting
df_clean = df.drop(columns=['Well', 'DEPTH', 'KTIM', 'PERM_LAB'], axis=1) # Assuming 'Poço' was previously renamed to 'Well', and 'PROF' to 'DEPTH'

# Reorder columns for the pair plot display order
df_ord = df_clean[['GR', 'RHOB', 'NPHI', 'DT', 'PHIT', 'VUGGY_INDEX', 'POR_LAB']]


# --- 5. Generate Pair Plot ---

# Set seaborn style (MOVED BEFORE PLOT)
sns.set(font_scale=1.5) # Reduced font_scale slightly for better fit, adjusted font size is set by font_scale implicitly.

print("Generating Pair Plot...")
# The diag_kind="hist" is used to show a histogram on the diagonal
sns.pairplot(df_ord, diag_kind="hist")
plt.suptitle("Pair Plot of Filtered Petrophysical Data", y=1.02, fontsize=16)

# Display the plot
plt.show() 