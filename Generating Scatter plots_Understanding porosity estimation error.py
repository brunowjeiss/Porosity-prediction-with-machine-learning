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

# Libraries for metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

# --- 1. Define Paths and Constants ---
# Use relative paths for portability

DATA_DIR = 'data/processed'
SCATTERPLOT_DIR = 'reports/figures/scatterplots' # Suggested dedicated directory for output images

INPUT_FILENAME = 'df_consolidado_vuggyeq.csv'
INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILENAME)

OUTPUT_FILENAME = 'scatter_PHIT_consolidated.png'
OUTPUT_PATH = os.path.join(SCATTERPLOT_DIR, OUTPUT_FILENAME)

# REMOVED: os.chdir('C:/Users/User/Desktop/NOVO DOUTORADO/NOVO CODIGOS')

# --- 2. Load Data and Conversion ---

# Load data
try:
    df_consolidado = pd.read_csv(INPUT_PATH)
    print(f"Data loaded successfully from {INPUT_PATH}")
except FileNotFoundError:
    print(f"ERROR: Consolidated file not found at {INPUT_PATH}. Check your data path.")
    exit()

# Convert NMR PHIT scale to %
df_consolidado["PHIT"] = 100 * df_consolidado["PHIT"]


# --- 3. Consolidated PHIT vs. POR LAB Plot ---

plt.style.use('bmh')
fig, ax = plt.subplots(figsize=(8, 8)) # Added figsize for better visualization

# Set up the scatter plot (Colored by VUGGY_INDEX)
plt.scatter(x='POR_LAB', y='PHIT', data=df_consolidado, c='VUGGY_INDEX', vmin=0, vmax=0.5, cmap='rainbow')

# Change the X and Y ranges
plt.xlim(0, 30)
plt.ylim(0, 30)

# Add 1:1 line (Best Practice for X vs Y plots)
plt.plot([0, 30], [0, 30], color='gray', linestyle='--', alpha=0.6, label='1:1 Line')

# Add in labels for the axes
plt.ylabel('PHIT (%)', fontsize=14)
plt.xlabel('POR LAB (%)', fontsize=14)

# Plot regression line
x = df_consolidado["PHIT"]
y = df_consolidado["POR_LAB"]

# Obtain m (slope) and b (intercept) of linear regression line
m, b = np.polyfit(x, y, 1)

# Add linear regression line to scatterplot
plt.plot(x, m*x+b, color='darkred', label=f'Fit Line (y={m:.2f}x + {b:.2f})')

# Colorbar
plt.colorbar(label='VUGGY INDEX')

# Calculate and display metrics
# Plotting R2 (Pearson correlation squared)
correlation_matrix = np.corrcoef(x, y)
r_squared = round(correlation_matrix[0,1]**2, 2)
print(f"R-squared: {r_squared}")

# Mean squared error (RMSE)
rms = round(sqrt(mean_squared_error(y, x)), 2)
print(f"RMSE: {rms}")

# Place a text box in upper left in axes coords
ax.text(0.05, 0.95, 'R\u00b2 = ' + str(r_squared), transform=ax.transAxes, fontsize=14, verticalalignment='top')
ax.text(0.05, 0.85, 'RMSE = ' + str(rms), transform=ax.transAxes, fontsize=14, verticalalignment='top')

# Displaying the title
plt.title('PHIT from NMR (Consolidated Data)')
plt.legend()

# Ensure save directory exists and save using relative path
os.makedirs(SCATTERPLOT_DIR, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"Plot saved to {OUTPUT_PATH}")
plt.show()




