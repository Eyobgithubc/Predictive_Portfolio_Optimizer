import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt


def train_arima(train_data, order):
    model = ARIMA(train_data, order=order)
    arima_model = model.fit()
    return arima_model


def train_sarima(train_data, order, seasonal_order):
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    sarima_model = model.fit()
    return sarima_model


def train_lstm(train_data, n_steps=50, n_features=1):
    
    X, y = [], []
    for i in range(n_steps, len(train_data)):
        X.append(train_data[i - n_steps:i])
        y.append(train_data[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], n_features))

 
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=20, batch_size=32, verbose=1)
    return model, n_steps


def forecast_lstm(model, test_data, n_steps=50):
    predictions = []
    for i in range(n_steps, len(test_data)):
        input_data = test_data[i - n_steps:i]
        input_data = input_data.reshape((1, n_steps, 1))
        predictions.append(model.predict(input_data, verbose=0)[0][0])
    return predictions


def evaluate_forecast(test_data, predictions):
   
    test_data = np.array(test_data)
    predictions = np.array(predictions)
    mae = mean_absolute_error(test_data, predictions)
    rmse = sqrt(mean_squared_error(test_data, predictions))
    mape = np.mean(np.abs((test_data - predictions) / test_data[test_data != 0])) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def forecast_arima(model, steps, start_index, test_index):

    forecast = model.forecast(steps=steps)
    forecast = pd.Series(forecast, index=test_index)
    return forecast
