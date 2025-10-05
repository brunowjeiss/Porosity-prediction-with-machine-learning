import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration for Portability ---
# Assuming the result CSV from the prediction step and the original data are available
BLIND_TEST_WELL_ID = 'well1'
# NOTE: The plotting assumes 'final_RF' is the file saved with predictions. 
# We'll need the path to the original full data (to get POÇO) and the predicted data.
FULL_DATA_PATH = 'data/DATAFRAMES DE POÇOS/df_consolidado.csv'
PREDICTED_DATA_PATH = 'results/blind_test_results.csv' # Assuming a standard result name

# --- 1. Load Data Required for Visualization ---

# Load full original data to extract the comparison data (POÇO)
try:
    df_full = pd.read_csv(FULL_DATA_PATH)
    # Extract original lab data for the specific well
    POCO = df_full[df_full['Poço'] == BLIND_TEST_WELL_ID].copy()
    if 'PROF' in POCO.columns:
        POCO.set_index('PROF', inplace=True)
except FileNotFoundError:
    print(f"Error: Original data not found at '{FULL_DATA_PATH}'. Cannot plot POR_LAB.")
    POCO = pd.DataFrame() # Create empty DF to prevent failure

# Load the predicted results for the log curves and predictions
try:
    # Use a standard predicted output name for generic visualization
    final_RF = pd.read_csv(PREDICTED_DATA_PATH)
    if 'PROF' in final_RF.columns:
        final_RF.set_index('PROF', inplace=True)
except FileNotFoundError:
    print(f"Error: Predicted data not found at '{PREDICTED_DATA_PATH}'. Visualization aborted.")
    exit()

# Set depth limits (Example values, use data min/max in production)
topo = final_RF.index.min() # Top of the log
base = final_RF.index.max() # Base of the log

# --- 2. Visualization (Log Plot) ---
fig, axes = plt.subplots(figsize=(10, 10))

# Curve names for plot titles
curve_names = ['Gamma', 'PHIT', 'VUGGY INDEX','POR PRED', 'PHIT', "POR PRED"]

# Set up the plot axes
ax1 = plt.subplot2grid((1,5), (0,0), rowspan=1, colspan = 1)
ax2 = plt.subplot2grid((1,5), (0,1), rowspan=1, colspan = 1)
ax3 = plt.subplot2grid((1,5), (0,2), rowspan=1, colspan = 1)
ax4 = plt.subplot2grid((1,5), (0,3), rowspan=1, colspan = 1)
ax5 = plt.subplot2grid((1,5), (0,4), rowspan=1, colspan = 1)
ax6 = ax5.twiny() # Twin for predicted porosity curve
ax7 = ax5.twiny() # Twin for laboratory points (POR_LAB)

# Set up the individual log tracks / subplots
ax1.plot(final_RF["GR"], final_RF.index, color = "green", lw = 0.5)
ax1.set_xlim(0, 200)
ax1.spines['top'].set_edgecolor('green')

ax2.plot(final_RF["PHIT"], final_RF.index, color = "red", lw = 0.5)
ax2.set_xlim(0.3,0.01 )
ax2.spines['top'].set_edgecolor('blue')

ax3.plot(final_RF["VUGGY_INDEX"], final_RF.index, color = "green", lw = 0.5)
ax3.set_xlim(1, 0)
ax3.spines['top'].set_edgecolor('green')

ax4.plot(final_RF["POR_PRED"], final_RF.index, color = "blue", lw = 0.5)
ax4.set_xlim(30, 0)
ax4.spines['top'].set_edgecolor('red')

ax5.plot(final_RF["PHIT"], final_RF.index, color = "red", lw = 0.5)
ax5.set_xlim(0.3, 0.01)
ax5.spines['top'].set_edgecolor('blue')

ax6.plot(final_RF["POR_PRED"], final_RF.index, color = "blue", lw = 0.5)
ax6.set_xlim(30, 0)
ax6.spines['bottom'].set_edgecolor('red') # Note: ax6 spine is set to bottom in original code, but top in the visualization logic loop

# ax7 plots POR_LAB from the POCO DataFrame
if not POCO.empty and 'POR_LAB' in POCO.columns:
    ax7.plot(POCO["POR_LAB"], POCO.index, color = "blue", marker="o", lw=0, markersize=3)
ax7.set_xlim(30, 0)
ax7.spines['top'].set_edgecolor('red')

# Set up the common elements between the subplots
for i, ax in enumerate(fig.axes):
    ax.set_ylim(base, topo) # Set the depth range (base to top)
    
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    
    # Adjust labeling based on axis index
    if i < 4:
        ax.set_xlabel(curve_names[i])
    elif i == 4:
        ax.set_xlabel(curve_names[4]) # PHIT for ax5
        ax.spines["top"].set_position(("axes", 1.0))
    elif i == 5:
        ax.set_xlabel(curve_names[5]) # POR PRED for ax6
        # Original code used (axes, 1.08) for ax5's twin (ax6)
        ax.spines["top"].set_position(("axes", 1.08))
    elif i == 6:
        # ax7 is another twin, sharing the POR PRED axis label space
        ax.spines["top"].set_position(("axes", 1.08))
    
    if i < 5: # Grid for axes 1 to 5 (index 0 to 4)
        ax.grid()
    
# Hide tick labels on the y-axis 
for ax in [ax2, ax3, ax4, ax5, ax6, ax7]:
    plt.setp(ax.get_yticklabels(), visible = False)

# Reduce the space between each subplot
fig.subplots_adjust(wspace = 0.08)

plt.savefig('logplot.png', dpi=150)
print("Log plot saved as logplot.png")