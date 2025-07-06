from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Load model and scaler with error handling
model = None
scaler = None

try:
    model = joblib.load("models/best_rain_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    print("Warning: Model files not found. Please ensure 'models/best_rain_model.pkl' and 'models/scaler.pkl' exist.")
    print("You can create these files by training a model and saving it with joblib.dump()")

# Default values for non-user inputs
df = pd.read_csv("traindata.csv")
selected_features = [
    'temp', 'humidity', 'sealevelpressure', 'cloudcover',
    'windspeed', 'dew', 'windgust', 'visibility'
]
default_values = {}
for feature in selected_features:
    default_values[feature] = df[feature].mean()

# FastAPI App
app = FastAPI(title="Rain Prediction API", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Request model
class WeatherInput(BaseModel):
    temp: float
    humidity: float
    dew: float

# Response model
class PredictionOutput(BaseModel):
    rain_probability: float
    will_rain: bool
    message: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "🌦️ Rain Prediction API is running!"}

# Predict endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict_weather(data: WeatherInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure model files exist.")
    
    input_data = pd.DataFrame({
        'temp': [data.temp],
        'humidity': [data.humidity],
        'sealevelpressure': [default_values['sealevelpressure']],
        'cloudcover': [default_values['cloudcover']],
        'windspeed': [default_values['windspeed']],
        'dew': [data.dew],
        'windgust': [default_values['windgust']],
        'visibility': [default_values['visibility']]
    })

    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    return {
        "rain_probability": round(prob, 4),
        "will_rain": bool(prediction),
        "message": "☔ Likely to Rain Today" if prob >= 0.6 else "🌤️ Not Likely to Rain Today"
    }
