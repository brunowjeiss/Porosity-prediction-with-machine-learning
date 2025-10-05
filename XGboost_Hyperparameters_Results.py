import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb # Import XGBoost library

# --- Configuration (Assumed from previous steps) ---
DATA_FILE_PATH = 'data/DATAFRAMES DE POÇOS/df_consolidado_vuggyeq.csv'
OUTPUT_DIR = 'results/test_data_predictions'
RANDOM_STATE = 30
SPLIT_STATE = 20
WELLS_TO_TRAIN = ['9-BRSA-716-RJS', '3-BRSA-496-RJS', '3-BRSA-795-RJS', '3-BRSA-755A-RJS'] # Actual well IDs
FEATURE_NAMES = ['GR', 'RHOB', 'NPHI', 'DT', 'PHIT', 'VUGGY_INDEX']
TARGET_NAME = 'POR_LAB'
OUTLIER_THRESHOLD = 5

# --- 1. Data Loading, Cleaning, and Preparation ---
try:
    df = pd.read_csv(DATA_FILE_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at '{DATA_FILE_PATH}'.")
    exit()

# Subset and clean data
df = df[df['Poço'].isin(WELLS_TO_TRAIN)].copy()
df['DIF_PHIT'] = df[TARGET_NAME] - (df['PHIT'] * 100)
df['DIF_VUGGY'] = df[TARGET_NAME] - (df['VUGGY_INDEX'] * 100)
df = df[
    (df["DIF_PHIT"].between(-OUTLIER_THRESHOLD, OUTLIER_THRESHOLD)) &
    (df["DIF_VUGGY"].between(-OUTLIER_THRESHOLD, OUTLIER_THRESHOLD))
].drop(columns=['DIF_PHIT', 'DIF_VUGGY']).copy()

# Split features (X) and target (y)
X = df[FEATURE_NAMES].copy()
y = df[TARGET_NAME].copy()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SPLIT_STATE
)

# Scaling (StandardScaler is fine, but XGBoost often performs well without it, 
# although scaling generally aids consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 2. XGBoost Training and Evaluation ---

# XGBoost Hyperparameters for Regression
# NOTE: n_estimators is reduced from 2000 (used in RF) as XGBoost typically converges faster.
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # Standard objective for mean squared error
    n_estimators=1000,             # Number of boosting rounds (trees)
    learning_rate=0.05,            # Step size shrinkage to prevent overfitting
    max_depth=6,                   # Max depth of a tree (controls model complexity)
    subsample=0.8,                 # Subsample ratio of the training instance (for stability)
    colsample_bytree=0.8,          # Subsample ratio of columns when constructing each tree
    gamma=0,                       # Minimum loss reduction required to make a further partition
    reg_alpha=0.005,               # L1 regularization term on weights (Lasso)
    random_state=RANDOM_STATE,
    n_jobs=-1                      # Use all available cores
)

# Fit the model to the scaled training data
xgb_model.fit(X_train_scaled, y_train)

# 3. Evaluation on Test Data
y_pred_XGB = xgb_model.predict(X_test_scaled)
mse_XGB = mean_squared_error(y_test, y_pred_XGB)
mae_XGB = mean_absolute_error(y_test, y_pred_XGB)
r2_XGB = r2_score(y_test, y_pred_XGB)

print('\n--- XGBoost (Hyperparameterized) Results ---')
print(f'Mean Squared Error (MSE): {mse_XGB:.4f}')
print(f'Mean Absolute Error (MAE): {mae_XGB:.4f}')
print(f'R2 Score: {r2_XGB:.4f}')

# 4. Training RMSE
y_train_pred_XGB = xgb_model.predict(X_train_scaled)
mse_train_XGB = mean_squared_error(y_train, y_train_pred_XGB)
rmse_train_XGB = round(np.sqrt(mse_train_XGB), 2)
print(f"Training RMSE: {rmse_train_XGB:.2f}")

# 5. Cross-Validation
scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=10, scoring='r2', n_jobs=-1)
print("\nCross-Validation R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 6. Save Results
# Prepare test results for merging
X_test_results = X_test.copy()
X_test_results['POR_PRED'] = y_pred_XGB
X_test_results['POR_LAB_TEST'] = y_test

# Merge with original data (df) to get Poço and PROF
inner_XGB = pd.merge(df, X_test_results, how='inner', on=FEATURE_NAMES, suffixes=('_orig', '_test'))
inner_XGB = inner_XGB.drop(columns=[TARGET_NAME])

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save XGBoost results
inner_XGB.to_csv(os.path.join(OUTPUT_DIR, 'inner_XGB_hyper.csv'), index=False)
print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'inner_XGB_hyper.csv')}")