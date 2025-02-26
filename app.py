import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load data
data = pd.read_excel('steel-data.xlsx', sheet_name='Sheet2')

# Define independent and dependent variables
X = data[['Cu conc.', ' CE', 'Major Phase']]
y = data['YS']

# One-hot encoding for 'Major Phase'
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_phase = encoder.fit_transform(X[['Major Phase']])
encoded_phase_df = pd.DataFrame(encoded_phase, columns=encoder.get_feature_names_out(['Major Phase']))

# Concatenate encoded categorical and numerical features
X = pd.concat([X.drop('Major Phase', axis=1).reset_index(drop=True), encoded_phase_df.reset_index(drop=True)], axis=1)

# Standardize numerical features
scaler = StandardScaler()
X[['Cu conc.', ' CE']] = scaler.fit_transform(X[['Cu conc.', ' CE']])

# Apply Polynomial Features (Higher Degree for Better Interactions)
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Optimized Random Forest Model
rf_model = RandomForestRegressor(
    n_estimators=300,        # More trees for better learning
    max_depth=15,            # Allow deeper trees to capture patterns
    min_samples_split=10,    # Reduce overfitting
    min_samples_leaf=3,      # Ensure smoother predictions
    max_features='sqrt',     # Random subset selection for diversity
    random_state=42
)

# Train Model
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate
print("Improved Random Forest R2 Score:", r2_score(y_test, y_pred_rf))
print("Improved Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))

# --- Predicting for a new sample ---
test_data = pd.DataFrame({'Cu conc.': [2.23, 1.55, 0.58, 0.51, 0.55, 0.1, 2.23, 1.61, 0.23, 0.04, 1.46, 0.28, 1.68, 1.06, 1.64, 0.2, 1.0, 0.28, 0.26, 0.35], 
                          ' CE': [0.9173, 0.5643, 0.22, 0.7197, 0.17, 0.07, 0.9173, 0.494, 0.172, 0.0713, 0.4527, 0.5, 0.4518, 0.454, 0.526, 0.0607, 2.9667, 0.2953, 0.07, 0.4], 
                          'Major Phase': ['Martensitic', 'Austenitic', 'Martensitic', 'Martensitic+Austenitic', 'Ferritic+Pearlitic', 'Ferritic+Martensitic', 'Martensitic', 
                                          'Ferritic+Martensitic', 'Ferritic+Pearlitic', 'Ferritic', 'Austenitic', 'Ferritic+Pearlitic', 'Ferrtic+Pearlitic', 
                                          'Austenitic', 'Ferritic+Martensitic', 'Ferritic', 'Austenitic', 'Ferritic+Pearlitic', 'Austenitic', 'Ferritic+Pearlitic']})

# One-hot encode 'Major Phase' for the new sample
encoded_test_phase = encoder.transform(test_data[['Major Phase']])
encoded_test_phase_df = pd.DataFrame(encoded_test_phase, columns=encoder.get_feature_names_out(['Major Phase']))

# Concatenate with numeric features
test_data = pd.concat([test_data.drop('Major Phase', axis=1).reset_index(drop=True), encoded_test_phase_df.reset_index(drop=True)], axis=1)

# Ensure test_data has all features from training data
test_data = test_data.reindex(columns=X.columns, fill_value=0)  # This ensures missing one-hot features are filled with 0

# Apply polynomial feature transformation to the test data
test_data_poly = poly.transform(test_data)

# Predict Yield Strength
predicted_ys_rf = rf_model.predict(test_data_poly)
print("Predicted Yield Strength (Random Forest):", predicted_ys_rf)

import pickle
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(poly, open("poly.pkl", "wb"))
