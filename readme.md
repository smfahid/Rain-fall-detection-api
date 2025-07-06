# 🌦️ Rain Prediction API

A machine learning-powered weather prediction API that forecasts rain probability based on weather parameters. Built with FastAPI and scikit-learn.

## 📋 Features

- **Rain Probability Prediction**: Predicts the likelihood of rain based on weather conditions
- **RESTful API**: Clean FastAPI endpoints for easy integration
- **Machine Learning Model**: Uses Random Forest Classifier for accurate predictions
- **CORS Support**: Ready for frontend integration
- **Real-time Predictions**: Instant weather forecasts

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd sana-project
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if not already trained)

   ```bash
   python train_model.py
   ```

5. **Run the API server**
   ```bash
   uvicorn main:app --reload
   ```

The API will be available at `http://localhost:8000`

## 📊 API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### GET `/`

Health check endpoint

```bash
curl http://localhost:8000/
```

**Response:**

```json
{
  "message": "🌦️ Rain Prediction API is running!"
}
```

#### POST `/predict`

Predict rain probability based on weather data

**Request Body:**

```json
{
  "temp": 25.0, // Temperature in Celsius
  "humidity": 70.0, // Humidity percentage
  "dew": 18.0 // Dew point in Celsius
}
```

**Response:**

```json
{
  "rain_probability": 0.2345,
  "will_rain": false,
  "message": "🌤️ Not Likely to Rain Today"
}
```

### Example Usage

```python
import requests

# Make a prediction
data = {
    "temp": 22.5,
    "humidity": 85.0,
    "dew": 19.0
}

response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()

print(f"Rain Probability: {result['rain_probability']:.2%}")
print(f"Will it rain? {result['will_rain']}")
print(f"Message: {result['message']}")
```

## 🏗️ Project Structure

```
sana-project/
├── main.py              # FastAPI application
├── train_model.py       # Model training script
├── traindata.csv        # Training dataset
├── requirements.txt     # Python dependencies
├── models/              # Trained models
│   ├── best_rain_model.pkl
│   └── scaler.pkl
├── venv/                # Virtual environment
└── readme.md           # This file
```

## 🤖 Model Details

### Features Used

- **Temperature** (°C)
- **Humidity** (%)
- **Sea Level Pressure** (hPa)
- **Cloud Cover** (%)
- **Wind Speed** (km/h)
- **Dew Point** (°C)
- **Wind Gust** (km/h)
- **Visibility** (km)

### Algorithm

- **Random Forest Classifier** with 100 estimators
- **StandardScaler** for feature normalization
- **Train/Test Split**: 80/20 ratio

### Model Performance

The model achieves high accuracy in predicting rain based on weather conditions. Default values are used for features not provided in the API request.

## 🔧 Development

### Training a New Model

```bash
python train_model.py
```

This script will:

1. Generate synthetic weather data
2. Train a Random Forest model
3. Save the model and scaler to `models/` directory
4. Display model performance metrics

### Adding New Features

1. Modify the `selected_features` list in both `train_model.py` and `main.py`
2. Update the `WeatherInput` model in `main.py`
3. Retrain the model

## 🌐 Frontend Integration

The API includes CORS middleware configured for:

- `http://localhost:3000`
- `http://127.0.0.1:3000`

You can easily integrate this API with React, Vue, or any frontend framework.

## 📝 Environment Variables

No environment variables are required for basic usage. The API uses default values for missing weather parameters.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

If you encounter any issues:

1. Check that all dependencies are installed
2. Ensure the model files exist in the `models/` directory
3. Verify the API server is running on the correct port

---

**Happy Weather Forecasting! ☔🌤️**
