# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
from datetime import datetime
import pandas as pd

app = Flask(__name__, template_folder='templates', static_folder='Static')

# Load all models from the Models folder
MODELS_DIR = 'Models'
models = {}

for i in range(1, 8):
    try:
        with open(f'{MODELS_DIR}/logistic_model{i}.pkl', 'rb') as f:
            models[f'model{i}'] = pickle.load(f)
        with open(f'{MODELS_DIR}/scaler{i}.pkl', 'rb') as f:
            models[f'scaler{i}'] = pickle.load(f)
        with open(f'{MODELS_DIR}/encoder{i}.pkl', 'rb') as f:
            models[f'encoder{i}'] = pickle.load(f)
    except Exception as e:
        print(f"Error loading model {i}: {str(e)}")

def prepare_features(flight_data):
    """Prepare features for prediction based on flight data"""
    features = pd.DataFrame({
        'airline': [flight_data['airline']],
        'origin': [flight_data['fromCode']],
        'destination': [flight_data['toCode']],
        'scheduled_departure_hour': [flight_data['departureTime'].split(':')[0]],
        'scheduled_arrival_hour': [flight_data['arrivalTime'].split(':')[0]],
        'flight_number': [flight_data['flightNumber']],
        'day_of_week': [datetime.strptime(flight_data['date'], '%Y-%m-%d').weekday()],
        'month': [datetime.strptime(flight_data['date'], '%Y-%m-%d').month]
    })
    return features

def predict_delay(flight_data):
    """Predict delay using ensemble of models"""
    features = prepare_features(flight_data)
    
    # Encode categorical features
    encoded_features = features.copy()
    for i in range(1, 8):
        try:
            encoder = models[f'encoder{i}']
            for col in ['airline', 'origin', 'destination']:
                if col in encoder.classes_:
                    encoded_features[col] = encoder.transform(encoded_features[col])
        except Exception as e:
            print(f"Encoding error for model {i}: {str(e)}")
    
    # Scale features and make predictions
    predictions = []
    for i in range(1, 8):
        try:
            scaler = models[f'scaler{i}']
            model = models[f'model{i}']
            
            # Select only numerical features for scaling
            numerical_features = encoded_features.select_dtypes(include=['int64', 'float64'])
            scaled_features = scaler.transform(numerical_features)
            
            # Predict
            pred = model.predict_proba(scaled_features)[0][1]  # Probability of delay
            predictions.append(pred)
        except Exception as e:
            print(f"Prediction error for model {i}: {str(e)}")
    
    # Ensemble prediction (average probability)
    if predictions:
        avg_prob = np.mean(predictions)
        return avg_prob > 0.5, avg_prob  # Threshold at 0.5
    return False, 0.0

@app.route('/')
def home():
    return render_template('ui3.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        is_delayed, probability = predict_delay(data)
        return jsonify({
            'isDelayed': is_delayed,
            'probability': float(probability),
            'confidence': min(0.99, max(0.5, probability * 1.2))  # Adjusted confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search_flights', methods=['POST'])
def search_flights():
    # Simulated flight data - in production replace with actual API call
    data = request.json
    from_airport = data['from']
    to_airport = data['to']
    date = data['date']
    
    # Generate mock flights
    flights = []
    for i in range(5 + int(hash(from_airport + to_airport) % 5)):  # 5-10 flights
        flight = {
            'airline': ['AA', 'DL', 'UA', 'WN', 'B6'][i % 5],
            'flightNumber': f"{['AA', 'DL', 'UA', 'WN', 'B6'][i % 5]}{1000 + i}",
            'departureTime': f"{6 + i % 12}:{['00', '30'][i % 2]}",
            'arrivalTime': f"{7 + i % 12}:{['30', '00'][i % 2]}",
            'fromCode': from_airport[:3],
            'toCode': to_airport[:3],
            'date': date
        }
        flights.append(flight)
    
    return jsonify(flights)

if __name__ == '__main__':
    app.run(debug=True) 