import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR # Import Support Vector Regression

# --- Configuration (Assumed from previous steps) ---
DATA_FILE_PATH = 'data/DATAFRAMES DE POÇOS/df_consolidado_vuggyeq.csv'
OUTPUT_DIR = 'results/test_data_predictions'
RANDOM_STATE = 30
SPLIT_STATE = 20
WELLS_TO_TRAIN = ['9-BRSA-716-RJS', '3-BRSA-496-RJS', '3-BRSA-795-RJS', '3-BRSA-755A-RJS']
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

# Scaling (CRUCIAL for SVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 2. SVR Training and Evaluation ---

# SVR Hyperparameters for Regression
# C: Regularization parameter. A higher value means less regularization (more complex model).
# epsilon: Defines the tube where errors are acceptable.
svr_model = SVR(
    kernel='rbf',        # Radial Basis Function (common choice for non-linear data)
    C=10,                # Regularization strength
    epsilon=0.1,         # Width of the margin of tolerance
    gamma='scale'        # Kernel coefficient (automatically calculated from features)
)

# Fit the model to the scaled training data
svr_model.fit(X_train_scaled, y_train)

# 3. Evaluation on Test Data
y_pred_SVR = svr_model.predict(X_test_scaled)
mse_SVR = mean_squared_error(y_test, y_pred_SVR)
mae_SVR = mean_absolute_error(y_test, y_pred_SVR)
r2_SVR = r2_score(y_test, y_pred_SVR)

print('\n--- Support Vector Regression (SVR) Results ---')
print(f'Mean Squared Error (MSE): {mse_SVR:.4f}')
print(f'Mean Absolute Error (MAE): {mae_SVR:.4f}')
print(f'R2 Score: {r2_SVR:.4f}')

# 4. Training RMSE
y_train_pred_SVR = svr_model.predict(X_train_scaled)
mse_train_SVR = mean_squared_error(y_train, y_train_pred_SVR)
rmse_train_SVR = round(np.sqrt(mse_train_SVR), 2)
print(f"Training RMSE: {rmse_train_SVR:.2f}")

# 5. Cross-Validation
# Note: SVR can be slower than tree-based models, so cross-validation may take time.
scores = cross_val_score(svr_model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1) 
print("\nCross-Validation R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# 6. Save Results
# Prepare test results for merging
X_test_results = X_test.copy()
X_test_results['POR_PRED'] = y_pred_SVR
X_test_results['POR_LAB_TEST'] = y_test

# Merge with original data (df) to get Poço and PROF
inner_SVR = pd.merge(df, X_test_results, how='inner', on=FEATURE_NAMES, suffixes=('_orig', '_test'))
inner_SVR = inner_SVR.drop(columns=[TARGET_NAME])

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save SVR results
inner_SVR.to_csv(os.path.join(OUTPUT_DIR, 'inner_SVR_hyper.csv'), index=False)
print(f"\nResults saved to {os.path.join(OUTPUT_DIR, 'inner_SVR_hyper.csv')}")