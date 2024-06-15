from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import load

app = Flask(__name__)

# Function to load the model and scaler for a given index
def load_model_and_scaler(index_name):
    model_path = f'{index_name}_lstm.h5'
    scaler_path = 'scaler.pkl'

    model = tf.keras.models.load_model(model_path)
    scaler = load(scaler_path)
    
    return model, scaler

def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window + 1]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100. / (1. + rs)

    for i in range(window, len(prices)):
        delta = deltas[i - 1]  # The diff is 1 shorter
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

@app.route('/predict/<index_name>', methods=['POST'])
def predict(index_name):
    try:
        seq_length = 60
        # Load the model and scaler for the requested index
        model, scaler = load_model_and_scaler(index_name)
        
        # Get the past 60 days of close prices from the request
        request_data = request.get_json(force=True)
        last_sequence = np.array(request_data['last_sequence']).reshape(-1, 1)
        
        # Ensure the input data is in the correct format
        if last_sequence.shape[0] != seq_length:
            return jsonify({'error': 'Invalid input sequence length'}), 400
        
        print(last_sequence)
        # Fit the scaler on the last sequence data
        scaler.fit(last_sequence)
        
        # Scale the input data
        last_sequence_scaled = scaler.transform(last_sequence)
        
        # Perform the next 30 days prediction
        next_30_days = []
        for _ in range(30):
            next_day_pred = model.predict(last_sequence_scaled.reshape(1, seq_length, 1))
            next_30_days.append(next_day_pred[0, 0])
            last_sequence_scaled = np.append(last_sequence_scaled[1:], next_day_pred).reshape(-1, 1)
        
        # Inverse transform the predictions
        next_30_days = scaler.inverse_transform(np.array(next_30_days).reshape(-1, 1))
        
        # Create a date range for the next 30 days
        last_date = pd.Timestamp(request_data['last_date'])
        next_dates = pd.date_range(start=last_date, periods=30) + pd.DateOffset(days=1)
        
        # Calculate RSI
        close_prices = np.append(last_sequence.flatten(), next_30_days.flatten())
        rsi = calculate_rsi(close_prices, window=14)
        last_rsi = rsi[-1]
        
        # Determine the trend
        if last_rsi > 70:
            trend = "trend is very likely up"
        elif last_rsi > 50:
            trend = "trend is likely up"
        else:
            trend = "trend is likely down"
        
        # Prepare the response
        predictions = [{str(date.date()): float(price)} for date, price in zip(next_dates, next_30_days)]
        
        return jsonify({'predictions': predictions, 'rsi': float(last_rsi), 'trend': trend})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
