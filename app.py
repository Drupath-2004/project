import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.metrics import r2_score, accuracy_score

# Function to calculate anomaly probability based on rules
def calculate_anomaly_probability(row):
    score = 0
    if row['Engine Temperature (°C)'] > 100:
        score += 0.4
    if row['Brake Pad Thickness (mm)'] < 3:
        score += 0.4
    elif row['Brake Pad Thickness (mm)'] < 5:
        score += 0.2
    if row['Tire Pressure (PSI)'] < 30 or row['Tire Pressure (PSI)'] > 35:
        score += 0.2
    return score

# Function to determine anomaly status
def determine_anomaly_status(row, rf_prediction, anomaly_score):
    is_anomaly = (
        row['Engine Temperature (°C)'] > 100
        or row['Brake Pad Thickness (mm)'] < 3
        or row['Tire Pressure (PSI)'] < 28
        or row['Tire Pressure (PSI)'] > 36
        or anomaly_score > 0.3
        or rf_prediction == 1
    )
    return is_anomaly

# Function to calculate RUL
def calculate_rul(row, model_prediction):
    base_rul = float(model_prediction)
    maintenance_multipliers = {0: 1.0, 1: 0.8, 2: 0.9}
    multiplier = maintenance_multipliers.get(row['Maintenance Type Label'], 1.0)
    return max(0, min(365, base_rul * multiplier))

# Function to calculate health score
def calculate_health_score(data):
    score = 100
    engine_temp = data['Engine Temperature (°C)'].values[0]
    brake_thickness = data['Brake Pad Thickness (mm)'].values[0]
    tire_pressure = data['Tire Pressure (PSI)'].values[0]

    if engine_temp > 100:
        score -= 25
    if brake_thickness < 3:
        score -= 30
    elif brake_thickness < 5:
        score -= 15
    if tire_pressure < 30 or tire_pressure > 35:
        score -= 20
    return max(0, score)


# Load models and preprocessing tools
try:
    fnn_model = tf.keras.models.load_model("fnn_model.h5")
    rf_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    X_test = joblib.load("X_test.pkl")
    y_test_rul = joblib.load("y_test_rul.pkl")
    y_test_anomaly = joblib.load("y_test_anomaly.pkl")
except FileNotFoundError as e:
    st.error("File not found: " + str(e))
    st.stop()
except Exception as e:
    st.error("Unexpected error: " + str(e))
    st.stop()

# Feature configurations
features = {
    'Engine Temperature (°C)': {'min': 70, 'max': 150, 'critical': 100, 'optimal': 90},
    'Brake Pad Thickness (mm)': {'min': 1, 'max': 10, 'critical': 3, 'optimal': 7},
    'Tire Pressure (PSI)': {'min': 20, 'max': 50, 'critical': 35, 'optimal': 32},
    'Maintenance Type Label': {}
}

st.title("Predictive Maintenance Dashboard")
st.sidebar.title("System Parameters")
for feature, config in features.items():
    if feature != "Maintenance Type Label":
        st.sidebar.write(f"**{feature}**: Critical > {config['critical']} | Optimal: {config['optimal']}")

# Input sliders
col1, col2, col3 = st.columns(3)
engine_temp = col1.slider("Engine Temperature (°C)", 70.0, 150.0, 90.0, step=0.1)
brake_thickness = col2.slider("Brake Pad Thickness (mm)", 1.0, 10.0, 7.0, step=0.1)
tire_pressure = col3.slider("Tire Pressure (PSI)", 20.0, 50.0, 32.0, step=0.1)

maintenance_type = st.selectbox(
    "Maintenance Type",
    ["Routine Maintenance", "Component Replacement", "Repair"]
)

# Process inputs
maintenance_mapping = {"Routine Maintenance": 0, "Component Replacement": 1, "Repair": 2}
user_input = pd.DataFrame({
    "Engine Temperature (°C)": [engine_temp],
    "Brake Pad Thickness (mm)": [brake_thickness],
    "Tire Pressure (PSI)": [tire_pressure],
    "Maintenance Type Label": [maintenance_mapping[maintenance_type]]
})

# Scale inputs
scaled_input = scaler.transform(user_input)

# Predict RUL and anomaly
try:
    rul_prediction = fnn_model.predict(scaled_input)[0][0]
    rf_prediction = rf_model.predict(user_input)[0]
except Exception as e:
    st.error("Prediction error: " + str(e))
    st.stop()

anomaly_score = calculate_anomaly_probability(user_input.iloc[0])
rul_days = calculate_rul(user_input.iloc[0], rul_prediction)
health_score = calculate_health_score(user_input)
is_anomaly = determine_anomaly_status(user_input.iloc[0], rf_prediction, anomaly_score)

# Results
st.subheader("Analysis Results")
st.metric("Remaining Useful Life (Days)", f"{rul_days:.2f}")
st.metric("Health Score", f"{health_score}/100")
st.metric("Anomaly Status", "Anomaly Detected" if is_anomaly else "Normal")

# Maintenance Recommendations
st.subheader("Recommendations")
recommendations = []
if engine_temp > 100:
    recommendations.append("Check cooling system immediately.")
if brake_thickness < 3:
    recommendations.append("Replace brake pads urgently.")
elif brake_thickness < 5:
    recommendations.append("Plan for brake pad replacement soon.")
if tire_pressure < 30 or tire_pressure > 35:
    recommendations.append("Adjust tire pressure to optimal range.")
if rul_days < 30:
    recommendations.append("Schedule comprehensive maintenance within 30 days.")

for rec in recommendations:
    st.markdown(f"- {rec}")

# Footer with model performance metrics
st.markdown("---")
st.markdown("### Model Performance Metrics")
col1, col2 = st.columns(2)

# Calculate RUL predictions for test data
rul_test_predictions = fnn_model.predict(X_test).flatten()
r2_value = r2_score(y_test_rul, rul_test_predictions)

# Calculate RF anomaly detection accuracy
anomaly_test_predictions = rf_model.predict(X_test)
anomaly_accuracy = accuracy_score(y_test_anomaly, anomaly_test_predictions)

with col1:
    st.metric("RUL Prediction R²", f"{r2_value:.3f}")

with col2:
    st.metric("Anomaly Detection Accuracy", f"{anomaly_accuracy:.1%}")
