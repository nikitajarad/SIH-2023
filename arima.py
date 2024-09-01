import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

# Generate example data
start_date = datetime(2017, 1, 1)
end_date = datetime(2023, 12, 31)
num_days = (end_date - start_date).days + 1
dates = [start_date + timedelta(days=i) for i in range(num_days)]

num_entries = len(dates)

# Generate synthetic data for each parameter
draft = np.random.uniform(5, 15, num_entries)
tidal_variation = np.random.uniform(0, 2, num_entries)
wave_height = np.random.uniform(0, 3, num_entries)
wave_period = np.random.uniform(5, 15, num_entries)
wave_direction = np.random.uniform(0, 360, num_entries)
current_speed = np.random.uniform(0.1, 2.5, num_entries)
flow_direction = np.random.uniform(0, 360, num_entries)
dredging_operations = np.random.choice([0, 1], size=num_entries, p=[0.8, 0.2])  # 20% chance of dredging
rainfall = np.random.uniform(0, 50, num_entries)
wind_speed = np.random.uniform(0, 20, num_entries)
temperature = np.random.uniform(10, 30, num_entries)
vessel_frequency = np.random.randint(1, 10, num_entries)
heavy_rainfall_events = np.random.choice([0, 1], size=num_entries, p=[0.9, 0.1])  # 10% chance of heavy rainfall
historical_siltation = np.random.uniform(1, 3, num_entries)
bathymetry = np.random.uniform(8, 20, num_entries)
shoreline_conditions = np.random.uniform(0, 1, num_entries)  # Example: vegetation or erosion index

# Assuming 'Date' is the datetime index
data = pd.DataFrame({
    'Date': dates,
    'Draft': draft,
    'Tidal_Variation': tidal_variation,
    'Wave_Height': wave_height,
    'Wave_Period': wave_period,
    'Wave_Direction': wave_direction,
    'Current_Speed': current_speed,
    'Flow_Direction': flow_direction,
    'Rainfall': rainfall,
    'Wind_Speed': wind_speed,
    'Temperature': temperature,
    'Vessel_Frequency': vessel_frequency,
    'Heavy_Rainfall_Events': heavy_rainfall_events,
    'Historical_Siltation': historical_siltation,
    'Bathymetry': bathymetry,
    'Shoreline_Conditions': shoreline_conditions,
    'Dredging_Operations': dredging_operations
})

# Set 'Date' as the datetime index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Use 'Dredging_Operations' as the target variable
y = data['Dredging_Operations']

# Split the data into training and testing sets
train_size = int(len(y) * 0.8)
train, test = y[0:train_size], y[train_size:]

# Fit ARIMA model
arima_model = ARIMA(train, order=(5, 1, 0))
arima_result = arima_model.fit()

# Forecast future values
forecast_steps = len(test)
forecast = arima_result.forecast(steps=forecast_steps)

# Plot original vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title('ARIMA Forecasting for Dredging_Operations')
plt.xlabel('Date')
plt.ylabel('Dredging_Operations')
plt.savefig('static/line_chart.jpeg')  # Save as PNG


# Generate Pie Chart
plt.figure(figsize=(8, 8))
pie_data = data['Dredging_Operations'].value_counts()
labels = pie_data.index
plt.pie(pie_data, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue'])
plt.title('Dredging Operations Distribution')
plt.savefig('static/pie_chart.jpeg', format='jpeg')  # Save as JPEG

# Generate Line Chart
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Wave_Height'], label='Wave Height')
plt.plot(data.index, data['Rainfall'], label='Rainfall')
plt.xlabel('Entries')
plt.ylabel('Values')
plt.title('Line Chart - Wave Height and Rainfall')
plt.legend()
plt.savefig('static/line_chart.jpeg', format='jpeg')  # Save as JPEG


# Generate Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Wind_Speed'], data['Temperature'], c=data['Dredging_Operations'], cmap='viridis')
plt.xlabel('Wind Speed')
plt.ylabel('Temperature')
plt.title('Scatter Plot - Wind Speed vs Temperature')
plt.colorbar(label='Dredging_Operations')
plt.savefig('static/scatter_plot.png')  # Save as PNG


# Generate Bar Chart
plt.figure(figsize=(10, 6))
bar_data = data['Vessel_Frequency'].value_counts().sort_index()
plt.bar(bar_data.index, bar_data)
plt.xlabel('Vessel Frequency')
plt.ylabel('Count')
plt.title('Bar Chart - Vessel Frequency Distribution')
plt.savefig('static/bar_chart.png')  # Save as PNG


# Generate Histogram
plt.figure(figsize=(10, 6))
plt.hist(data['Historical_Siltation'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Historical Siltation')
plt.ylabel('Frequency')
plt.title('Histogram - Historical Siltation Distribution')
plt.savefig('static/histogram.png')  # Save as PNG


# Show all plots
plt.show()

# ... (Remaining code)

# Convert forecast to binary using a threshold (e.g., 0.5)
binary_forecast = (forecast >= 0.5).astype(int)

# Calculate accuracy score
accuracy = accuracy_score(test, binary_forecast)
print(f'Testing Accuracy Score: {accuracy}')



# Forecast future values for training data
forecast_train = arima_result.forecast(steps=len(train))

# Convert forecast to binary using a threshold (e.g., 0.5) for training data
binary_forecast_train = (forecast_train >= 0.5).astype(int)

# Calculate accuracy score for training data
accuracy_train = accuracy_score(train, binary_forecast_train)
print(f'Training Accuracy Score: {accuracy_train}')


with open("arima_model.pkl", 'wb') as model_file:
    pickle.dump(arima_result, model_file)
    