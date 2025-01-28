import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load data
data = pd.read_csv('cars_hyundai_one_hot_encoded.csv')

# Check for NaN values and ensure RUL column is not all zeros
if data['Adjusted_RUL'].isna().any():
    raise ValueError("Adjusted_RUL column contains NaN values.")
if data['Adjusted_RUL'].sum() == 0:
    raise ValueError("Adjusted_RUL column contains all zeros.")

# Selected features and target
numerical_features = [
    'Engine Temperature (°C)',
    'Brake Pad Thickness (mm)',
    'Tire Pressure (PSI)'
]
categorical_feature = 'Maintenance Type Label'
X_numerical = data[numerical_features]
X_categorical = data[categorical_feature]
y_rul = data['Adjusted_RUL']
y_anomaly = data['Anomaly Indication']

# Encode categorical feature
label_encoder = LabelEncoder()
X_categorical_encoded = label_encoder.fit_transform(X_categorical)

# Combine numerical and categorical features
X = pd.concat([X_numerical, pd.DataFrame(X_categorical_encoded, columns=[categorical_feature])], axis=1)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train_rul, y_test_rul = train_test_split(X_scaled, y_rul, test_size=0.2, random_state=42)
_, _, y_train_anomaly, y_test_anomaly = train_test_split(X_scaled, y_anomaly, test_size=0.2, random_state=42)

# Train Random Forest for Anomaly Detection
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=8,
    random_state=42
)
rf_model.fit(X_train, y_train_anomaly)

# Build Fully Connected Neural Network (FNN) for RUL Prediction
fnn_model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Output layer for regression
])

# Compile the FNN model
fnn_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train FNN model
history = fnn_model.fit(
    X_train,
    y_train_rul,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the FNN model
y_pred_rul = fnn_model.predict(X_test).flatten()
mse = mean_squared_error(y_test_rul, y_pred_rul)
mae = mean_absolute_error(y_test_rul, y_pred_rul)

print(f"FNN - MSE: {mse:.2f}, MAE: {mae:.2f}")

# Predict and evaluate Random Forest model
y_pred_anomaly = rf_model.predict(X_test)
anomaly_accuracy = accuracy_score(y_test_anomaly, y_pred_anomaly)

print(f"Random Forest - Anomaly Detection Accuracy: {anomaly_accuracy:.2%}")

# Save models and preprocessing tools
fnn_model.save("fnn_model.h5")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test_rul, "y_test_rul.pkl")
joblib.dump(y_test_anomaly, "y_test_anomaly.pkl")

print("Models and data saved successfully.")