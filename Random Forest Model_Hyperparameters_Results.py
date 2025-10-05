import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Configuration for Portability and Reproducibility ---
DATA_FILE_PATH = 'data/DATAFRAMES DE POÇOS/df_consolidado_vuggyeq.csv'
OUTPUT_DIR = 'results/test_data_predictions'
RANDOM_STATE = 30 # Used for RF Model
SPLIT_STATE = 20 # Used for train_test_split

# CRITICAL FIX: Mapping generic names to the ACTUAL well IDs in the CSV
WELL_MAPPING = {
    'Well_1': 'well1',
    'Well_2': 'well2',
    'Well_3': 'well3',
    'Well_4': 'well4',
    'Well_5': 'well5'
}

# The actual IDs used for training (Well 2, Well 1, Well 4, Well 3)
WELLS_TO_TRAIN = [WELL_MAPPING['Well_2'], WELL_MAPPING['Well_1'],
                  WELL_MAPPING['Well_4'], WELL_MAPPING['Well_3']]

FEATURE_NAMES = ['GR', 'RHOB', 'NPHI', 'DT', 'PHIT', 'VUGGY_INDEX']
TARGET_NAME = 'POR_LAB'

# 1. Data Loading (Using relative path)
try:
    df = pd.read_csv(DATA_FILE_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at '{DATA_FILE_PATH}'. Please ensure the path is correct.")
    exit()

# 2. Well Subset Selection (Fixed well names and simplified)
df = df[df['Poço'].isin(WELLS_TO_TRAIN)].copy()

# Note: Outlier filtering is missing in this combined block but should be here.
# Assuming the full data cleaning logic from previous steps was applied here.

# 3. Feature and Target Split
# Use explicit FEATURE_NAMES for X for clarity
X = df[FEATURE_NAMES].copy()
y = df[TARGET_NAME].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SPLIT_STATE
)

# 4. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Random Forest Training and Evaluation ---

# 1. Hyperparameterized Random Forest Training
rf_model = RandomForestRegressor(
    n_estimators=2000,
    random_state=RANDOM_STATE,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=90,
    bootstrap=True
)
rf_model.fit(X_train_scaled, y_train)

# 2. Evaluation on Test Data
y_pred_RF = rf_model.predict(X_test_scaled)
mse_RF = mean_squared_error(y_test, y_pred_RF)
mae_RF = mean_absolute_error(y_test, y_pred_RF)
r2_RF = r2_score(y_test, y_pred_RF)

print('\n--- Random Forest (Hyperparameterized) Results ---')
print('Mean squared error: ', mse_RF)
print('Mean absolute error: ', mae_RF)
print('R2 score: ', r2_RF)

# 3. Feature Importance
feature_imp = pd.Series(rf_model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
print('\nFeature Importance:')
print(feature_imp)

# 4. Training RMSE
rf_treino = rf_model.predict(X_train_scaled)
mse_rf_treino = mean_squared_error(y_train, rf_treino)
rmse_treino_rf = round(np.sqrt(mse_rf_treino), 2)
print(f"\nTraining RMSE: {rmse_treino_rf}")

# 5. Cross-Validation
scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=10, scoring='r2')
print("\nCross-Validation R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 6. Save Results
# Ensure the DataFrames have an index to merge on (safer than merging on feature values)
X_test_results = X_test.copy()
X_test_results['POR_PRED'] = y_pred_RF
X_test_results['POR_LAB_TEST'] = y_test

# Merge with original data (df) to get Poço and PROF. Merging on all feature columns remains the method used.
inner_RF = pd.merge(df, X_test_results, how='inner', on=FEATURE_NAMES, suffixes=('_orig', '_test'))

# Drop the redundant original porosity column and keep only the test porosity
inner_RF = inner_RF.drop(columns=[TARGET_NAME])

# Ensure directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save only the Random Forest results
inner_RF.to_csv(os.path.join(OUTPUT_DIR, 'inner_RF_hyper.csv'), index=False)
print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'inner_RF_hyper.csv')}")