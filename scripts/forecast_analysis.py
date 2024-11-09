import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def load_lstm_model(pkl_file_path):
   
    with open(pkl_file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def prepare_data_for_lstm(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=sequence_length, batch_size=32)
    return generator, scaler



def generate_forecast(model, data, forecast_steps):
    
    forecast = []
    input_data = data[-1:]  
    
 
    input_data = np.array(input_data)
    
    for _ in range(forecast_steps):
      
        input_data_reshaped = input_data.reshape((1, 1, 1)) 
   
        predicted_value = model.predict(input_data_reshaped)
  
        forecast.append(predicted_value[0, 0])

        input_data = np.append(input_data, predicted_value)[-1:]

    return np.array(forecast)


def calculate_confidence_intervals(forecast, confidence_interval=0.10):
    
    forecast_upper = forecast * (1 + confidence_interval)
    forecast_lower = forecast * (1 - confidence_interval)
    return forecast_upper, forecast_lower

def visualize_forecast(original_data, forecast_data):
   
    plt.figure(figsize=(10, 6))
    plt.plot(original_data, label='Original Data', color='blue')
    plt.plot(range(len(original_data), len(original_data) + len(forecast_data)), forecast_data, 
             label='Forecast', color='red', linestyle='--')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Original Data vs. Forecast')
    plt.show()

def analyze_trend(forecast):
   

    trend_direction = "upward" if forecast[-1] > forecast[0] else "downward" if forecast[-1] < forecast[0] else "stable"
    
    return f"The forecasted trend is {trend_direction}."


def calculate_volatility(forecast, historical_data):
   
   
    forecast_series = pd.Series(forecast)
    historical_series = pd.Series(historical_data)
    
    
    forecast_volatility = forecast_series.rolling(window=30).std()  
    historical_volatility = historical_series.rolling(window=30).std()
    
    
    if forecast_volatility.mean() > historical_volatility.mean():
        volatility_analysis = "The forecast shows increased volatility compared to historical levels."
    else:
        volatility_analysis = "The forecast shows stable or decreased volatility compared to historical levels."
    
    return volatility_analysis, forecast_volatility.describe()



def assess_market_opportunities(trend_analysis, volatility_analysis):
  
    
    
    print(f"trend_analysis: {trend_analysis} (type: {type(trend_analysis)})")
    print(f"volatility_analysis: {volatility_analysis} (type: {type(volatility_analysis)})")
    
   
    if isinstance(trend_analysis, tuple):
        trend_analysis = trend_analysis[0] 
    
   
    if isinstance(trend_analysis, str):
        trend_contains_upward = "upward" in trend_analysis.lower()  
    else:
        trend_contains_upward = False

    
    if isinstance(volatility_analysis, tuple):
        print("volatility_analysis is a tuple.")
       
        if isinstance(volatility_analysis[0], str):
            volatility_analysis_str = volatility_analysis[0]
            print(f"volatility_analysis (string description): {volatility_analysis_str}")
            volatility_contains_decreased = "decreased volatility" in volatility_analysis_str.lower()
        elif isinstance(volatility_analysis[1], pd.Series):
            volatility_analysis_data = volatility_analysis[1]
            print(f"volatility_analysis (numerical data): {volatility_analysis_data}")
            volatility_mean = volatility_analysis_data.mean()
            volatility_contains_decreased = volatility_mean < 0.01  
        else:
            print(f"Unexpected format in volatility_analysis tuple: {volatility_analysis}")
            raise ValueError("volatility_analysis tuple is in an unexpected format.")
    
    elif isinstance(volatility_analysis, pd.Series):
        print("volatility_analysis is a pandas Series.")
       
        volatility_mean = volatility_analysis.mean()
        volatility_contains_decreased = volatility_mean < 0.01  
    elif isinstance(volatility_analysis, str):
        print("volatility_analysis is a string.")
        
        volatility_contains_decreased = "decreased volatility" in volatility_analysis.lower()
    else:
        print(f"Unexpected type for volatility_analysis: {type(volatility_analysis)}")
        raise ValueError(f"Unexpected type for volatility_analysis: {type(volatility_analysis)}")
    
    
    print(f"trend_contains_upward: {trend_contains_upward}")
    print(f"volatility_contains_decreased: {volatility_contains_decreased}")
    
    
    if trend_contains_upward and volatility_contains_decreased:
        return "There may be an opportunity for gains with lower risk in this period."
    elif "downward" in trend_analysis.lower() and "increased volatility" in volatility_analysis[0].lower():  # Fix here
        return "Potential for losses with higher risk; exercise caution in this period."
    else:
        return "The market shows mixed indicators; evaluate carefully before proceeding."
