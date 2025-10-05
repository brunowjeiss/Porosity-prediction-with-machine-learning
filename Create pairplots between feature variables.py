
# Libraries for data manipulation and plotting
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Define Paths and Constants ---

# Use relative paths for portability
DATA_DIR = 'data/processed'
SCATTERPLOT_DIR = 'reports/figures/scatterplots' # Directory for output images

INPUT_FILENAME = 'df_consolidado_vuggyeq.csv' # Assuming this file exists after prior processing
INPUT_PATH = os.path.join(DATA_DIR, INPUT_FILENAME)

OUTPUT_FILENAME = 'pairplot_full_data.png'
OUTPUT_PATH = os.path.join(SCATTERPLOT_DIR, OUTPUT_FILENAME)

# Ensure output directory exists
os.makedirs(SCATTERPLOT_DIR, exist_ok=True)

# --- 2. Load Data ---

# REMOVED: os.chdir(...)

print(f"Loading data from {INPUT_PATH}...")
try:
    df = pd.read_csv(INPUT_PATH)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Consolidated file not found at {INPUT_PATH}. Check your data path.")
    exit()

# --- 3. Prepare Data for Plotting ---

# Convert PHIT to percentage scale (as done in previous scripts)
df['PHIT'] = 100 * df['PHIT']

# Columns to include in the pair plot
# Assuming 'Poço' was previously renamed to 'Well', and 'PROF' to 'DEPTH'
COLUMNS_TO_PLOT = ['GR', 'RHOB', 'NPHI', 'DT', 'PHIT', 'VUGGY_INDEX', 'POR_LAB']

# Select and rename columns for clear visualization
df_plot = df[COLUMNS_TO_PLOT]

# Rename columns for plot clarity
df_renamed = df_plot.rename(columns={
    'POR_LAB': 'Laboratory Porosity (%)',
    'VUGGY_INDEX': 'Vuggy Index',
    'GR': 'Gamma Ray',
    'RHOB': 'Bulk Density',
    'NPHI': 'Neutron Porosity',
    'DT': 'Delta T (us/ft)',
    'PHIT': 'Total Porosity (NMR) (%)'
}, inplace=False)


# --- 4. Generate Pair Plot ---

# Set seaborn style and context
sns.set_theme(style="ticks", font_scale=1.2)

print("Generating Pair Plot for full dataset...")
# Create the pair plot
pairplot_fig = sns.pairplot(df_renamed, diag_kind="hist")

# Add a title to the figure
plt.suptitle("Pair Plot of Petrophysical Features (Full Dataset)", y=1.02, fontsize=16)

# Save the plot using a relative path
pairplot_fig.savefig(OUTPUT_PATH, dpi=150)
print(f"Plot saved to {OUTPUT_PATH}")

plt.show()