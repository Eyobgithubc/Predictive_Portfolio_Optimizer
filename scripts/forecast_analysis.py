import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def load_lstm_model(pkl_file_path):
    """
    Load the saved LSTM model from a pickle file.

    Parameters:
        pkl_file_path: str, path to the saved LSTM model (.pkl)

    Returns:
        model: Loaded LSTM model.
    """
    with open(pkl_file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def prepare_data_for_lstm(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)
    
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=sequence_length, batch_size=32)
    return generator, scaler

def generate_forecast(model, forecast_steps):
    """
    Generates a forecast using the loaded LSTM model.
    
    Parameters:
        model: Trained LSTM model.
        forecast_steps: int, the number of future time steps to forecast.
        
    Returns:
        forecast: np.array of forecasted values.
    """
    return model.predict(forecast_steps)

def calculate_confidence_intervals(forecast, confidence_interval=0.10):
    """
    Calculate upper and lower confidence intervals for the forecast.
    
    Parameters:
        forecast: np.array, forecasted values.
        confidence_interval: float, the percentage for confidence interval range (e.g., 0.10 for ±10%).
        
    Returns:
        forecast_upper: np.array, upper confidence interval values.
        forecast_lower: np.array, lower confidence interval values.
    """
    forecast_upper = forecast * (1 + confidence_interval)
    forecast_lower = forecast * (1 - confidence_interval)
    return forecast_upper, forecast_lower

def visualize_forecast(historical_data, forecast, forecast_upper, forecast_lower, forecast_dates):
    """
    Plot historical and forecasted data with confidence intervals.
    
    Parameters:
        historical_data: pd.Series or pd.DataFrame, historical time series data.
        forecast: pd.Series, forecasted data.
        forecast_upper: pd.Series, upper confidence interval.
        forecast_lower: pd.Series, lower confidence interval.
        forecast_dates: pd.DatetimeIndex, dates for the forecasted values.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data.index, historical_data, label="Historical Data", color='blue')
    plt.plot(forecast.index, forecast, label="Forecasted Data", color="orange")
    plt.fill_between(forecast.index, forecast_lower, forecast_upper, color="gray", alpha=0.3, label="Confidence Interval")
    plt.legend()
    plt.title("Tesla Stock Price Forecast with Confidence Intervals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

def analyze_trend(forecast):
    """
    Analyze the trend of the forecast.
    
    Parameters:
        forecast: pd.Series, forecasted data.
        
    Returns:
        trend_analysis: str, description of the forecast trend direction.
    """
    trend_direction = "upward" if forecast.iloc[-1] > forecast.iloc[0] else "downward" if forecast.iloc[-1] < forecast.iloc[0] else "stable"
    return f"The forecasted trend is {trend_direction}."

def detect_anomalies(forecast, forecast_upper, forecast_lower):
    """
    Detect anomalies in the forecast where values exceed confidence intervals.
    
    Parameters:
        forecast: pd.Series, forecasted data.
        forecast_upper: pd.Series, upper confidence interval.
        forecast_lower: pd.Series, lower confidence interval.
        
    Returns:
        pattern_analysis: str, description of anomaly detection results.
    """
    anomalies = np.where((forecast > forecast_upper) | (forecast < forecast_lower))[0]
    if len(anomalies) > 0:
        return f"Anomalies detected at {len(anomalies)} points in the forecast range."
    else:
        return "No significant anomalies detected in the forecast."

def calculate_volatility(forecast, historical_data):
    """
    Calculate and compare forecast volatility with historical volatility.
    
    Parameters:
        forecast: pd.Series, forecasted data.
        historical_data: pd.Series or pd.DataFrame, historical time series data.
        
    Returns:
        volatility_analysis: str, description of volatility analysis.
        forecast_volatility_summary: pd.Series, summary statistics of forecast volatility.
    """
    forecast_volatility = forecast.rolling(window=30).std()  # 30-day rolling standard deviation
    historical_volatility = historical_data.rolling(window=30).std()
    
    if forecast_volatility.mean() > historical_volatility.mean():
        volatility_analysis = "The forecast shows increased volatility compared to historical levels."
    else:
        volatility_analysis = "The forecast shows stable or decreased volatility compared to historical levels."
    
    return volatility_analysis, forecast_volatility.describe()

def assess_market_opportunities(trend_analysis, volatility_analysis):
    """
    Assess potential market opportunities and risks based on trend and volatility analysis.
    
    Parameters:
        trend_analysis: str, description of the trend analysis.
        volatility_analysis: str, description of the volatility analysis.
        
    Returns:
        market_assessment: str, description of potential market opportunities and risks.
    """
    if "upward" in trend_analysis and "decreased volatility" in volatility_analysis:
        return "There may be an opportunity for gains with lower risk in this period."
    elif "downward" in trend_analysis and "increased volatility" in volatility_analysis:
        return "Potential for losses with higher risk; exercise caution in this period."
    else:
        return "The market shows mixed indicators; evaluate carefully before proceeding."

def forecast_analysis_lstm(model_path, historical_data, forecast_steps=365, confidence_interval=0.10):
    """
    Main function to load the model, perform the forecast, and conduct analysis.
    
    Parameters:
        model_path: str, path to the saved LSTM model (.pkl).
        historical_data: pd.Series or pd.DataFrame, historical time series data.
        forecast_steps: int, number of future time steps to forecast.
        confidence_interval: float, confidence interval range (e.g., 0.10 for ±10%).
        
    Returns:
        insights: dict with analysis and interpretations of trends, volatility, and risks.
    """
    # Load model and generate forecast
    model = load_lstm_model(model_path)
    forecast = generate_forecast(model, forecast_steps)
    
    # Calculate confidence intervals
    forecast_upper, forecast_lower = calculate_confidence_intervals(forecast, confidence_interval)
    
    # Prepare data for plotting
    forecast_dates = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
    forecast = pd.Series(forecast, index=forecast_dates)
    forecast_upper = pd.Series(forecast_upper, index=forecast_dates)
    forecast_lower = pd.Series(forecast_lower, index=forecast_dates)
    
    # Visualization
    visualize_forecast(historical_data, forecast, forecast_upper, forecast_lower, forecast_dates)
    
    # Insights
    insights = {}
    insights["Trend Analysis"] = analyze_trend(forecast)
    insights["Pattern Analysis"] = detect_anomalies(forecast, forecast_upper, forecast_lower)
    volatility_analysis, forecast_volatility_summary = calculate_volatility(forecast, historical_data)
    insights["Volatility Analysis"] = volatility_analysis
    insights["Forecast Volatility Summary"] = forecast_volatility_summary
    insights["Market Opportunities and Risks"] = assess_market_opportunities(insights["Trend Analysis"], insights["Volatility Analysis"])
    
    return insights
