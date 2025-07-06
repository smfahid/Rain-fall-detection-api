import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Create sample training data
np.random.seed(42)
n_samples = 1000

# Generate realistic weather data
data = {
    'temp': np.random.normal(20, 10, n_samples),  # Temperature in Celsius
    'humidity': np.random.uniform(30, 100, n_samples),  # Humidity percentage
    'sealevelpressure': np.random.normal(1013, 10, n_samples),  # Pressure in hPa
    'cloudcover': np.random.uniform(0, 100, n_samples),  # Cloud cover percentage
    'windspeed': np.random.uniform(0, 30, n_samples),  # Wind speed in km/h
    'dew': np.random.normal(15, 8, n_samples),  # Dew point in Celsius
    'windgust': np.random.uniform(0, 50, n_samples),  # Wind gust in km/h
    'visibility': np.random.uniform(0, 20, n_samples),  # Visibility in km
}

df = pd.DataFrame(data)

# Create rain labels based on weather conditions
# Rain is more likely with high humidity, low temperature, high cloud cover, and low visibility
rain_probability = (
    (df['humidity'] / 100) * 0.4 +
    ((30 - df['temp']) / 30) * 0.3 +
    (df['cloudcover'] / 100) * 0.2 +
    ((20 - df['visibility']) / 20) * 0.1
)
rain_probability = np.clip(rain_probability, 0, 1)

# Add some randomness
rain_probability += np.random.normal(0, 0.1, n_samples)
rain_probability = np.clip(rain_probability, 0, 1)

df['rain'] = (rain_probability > 0.5).astype(int)

# Save training data
df.to_csv('traindata.csv', index=False)
print("Training data saved to traindata.csv")

# Prepare features and target
selected_features = [
    'temp', 'humidity', 'sealevelpressure', 'cloudcover',
    'windspeed', 'dew', 'windgust', 'visibility'
]
X = df[selected_features]
y = df['rain']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create models directory
os.makedirs('models', exist_ok=True)

# Save model and scaler
joblib.dump(model, 'models/best_rain_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\nModel and scaler saved to models/ directory")
print("You can now run the FastAPI server with: uvicorn main:app --reload") 