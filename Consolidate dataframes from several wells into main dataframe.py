
# Libraries for data manipulation and plotting (Active Imports)
import pandas as pd
import numpy as np 
import os

# --- 1. Define Paths and Constants ---

# Define the root directory for all input CSV files
DATA_DIR = 'data/raw_well_data'

# Define the output directory and filename for the consolidated data
OUTPUT_DIR = 'data/processed'
OUTPUT_FILENAME = 'df_consolidated.csv'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# Standard list of columns required for the final consolidated DataFrame
STANDARD_COLUMNS = ['Well', 'DEPTH', 'GR', 'RHOB', 'NPHI', 'DT', 'PHIT', 'KTIM', 'POR_LAB', 'PERM_LAB', 'VUGGY_INDEX']

# Define the relative paths to the input files
INPUT_FILES = {
    'well_1': os.path.join(DATA_DIR, 'well1.csv'),
    'well_2': os.path.join(DATA_DIR, 'well2.csv'),
    'well_3': os.path.join(DATA_DIR, 'well3.csv'),
    'well_4': os.path.join(DATA_DIR, 'well4.csv'),
    'well_5': os.path.join(DATA_DIR, 'well5.csv'),
}

# List of sequential well names for retrieval (Redundant but harmless)
SEQUENTIAL_WELL_NAMES = list(INPUT_FILES.keys())

# --- 2. Load CSV Files ---
# Create a dictionary to hold the loaded DataFrames
wells = {}

print("Loading and cleaning well data...")
for well_name, file_path in INPUT_FILES.items():
    try:
        df = pd.read_csv(file_path, sep=',')
        # Add the sequential 'Well' identifier column before storing/processing
        df['Well'] = well_name
        wells[well_name] = df
        print(f"Successfully loaded well {well_name} from {file_path}")
    except FileNotFoundError:
        print(f"ERROR: File not found for well {well_name} at {file_path}")


# Assign loaded DataFrames using the correct sequential keys
well_1 = wells.get('well_1')
well_2 = wells.get('well_2')
well_3 = wells.get('well_3')
well_4 = wells.get('well_4')
well_5 = wells.get('well_5')


# --- 3. Drop Unused Columns ---
# Check if all 5 wells loaded successfully using the correct variables
if all([well_1 is not None, well_2 is not None, well_3 is not None, well_4 is not None, well_5 is not None]):
    
    # --- Drop unused columns (cleaning specific to original well structure) ---
    well_1_clean = well_1.drop(['BS_EP', 'CALI', 'RD_EP', 'RM_EP', 'RS_EP', 'TCMR'], axis=1)
    well_2_clean = well_2.drop(['AE10', 'AE30', 'AE90', 'DEPTH_2'], axis=1)
    well_3_clean = well_3.drop(['CALI', 'RD_EP', 'RM_EP', 'RS_EP', 'RT10_EP', 'RT20_EP', 'RT30_EP',
                                'RT60_EP', 'RT90_EP'], axis=1)
    well_4_clean = well_4.drop(['RT10', 'RT30', 'RT90'], axis=1)
    well_5_clean = well_5.drop(['RD', 'RS'], axis=1)
    
    # --- 4. Rename Columns ---
    
    # Rename columns to match the standard
    renamed_1 = well_1_clean.rename(columns={'DTCO_EP': 'DT', 'PHIT_EP': 'PHIT', 'VUGGY_POR_INDEX_DEC2021': 'VUGGY_INDEX'}, inplace=False)
    renamed_2 = well_2_clean.rename(columns={'DTCO': 'DT'}, inplace=False)
    renamed_3 = well_3_clean.rename(columns={'RHOB_EP': 'RHOB', 'VUGGY_POR_INDEX': 'VUGGY_INDEX'}, inplace=False)
    renamed_4 = well_4_clean.rename(columns={'VUGGY_POR_INDEX': 'VUGGY_INDEX'}, inplace=False)
    renamed_5 = well_5_clean.rename(columns={'VUGGY_POR_INDEX': 'VUGGY_INDEX', 'DTCO': 'DT'}, inplace=False)
    
    
    # --- 5. Reorder Columns ---
    
    # Use the defined STANDARD_COLUMNS list for cleaner reordering
    well_1_ord = renamed_1[STANDARD_COLUMNS]
    well_2_ord = renamed_2[STANDARD_COLUMNS]
    well_3_ord = renamed_3[STANDARD_COLUMNS]
    well_4_ord = renamed_4[STANDARD_COLUMNS]
    well_5_ord = renamed_5[STANDARD_COLUMNS]
    
    
    # --- 6. Equalize Vuggy Index Means (Feature Engineering) ---
    # RE-INSERTED MISSING STEP
    print("Performing Vuggy Index equalization...")
    well_3_ord['VUGGY_INDEX'] = well_3_ord['VUGGY_INDEX'] * 0.7
    well_5_ord['VUGGY_INDEX'] = well_5_ord['VUGGY_INDEX'] * 0.5
    
    
    # --- 7. Perform Append (Concatenate DataFrames) ---
    df_consolidated = pd.concat([well_1_ord, well_2_ord, well_3_ord, well_4_ord, well_5_ord], ignore_index=True)
    
    print("Data consolidation complete.")
    
    # --- 8. Export Consolidated DataFrame ---
    
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Export DataFrame to the combined relative output path
    print(f"Exporting consolidated data to: {OUTPUT_PATH}")
    df_consolidated.to_csv(OUTPUT_PATH, index=False)
    print("Export complete.")

else:
    print("\nProcessing aborted due to missing input files.")




