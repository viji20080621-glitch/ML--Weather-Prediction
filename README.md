# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
Environmental conditions such as humidity, atmospheric pressure, wind speed, and seasonal variations affect temperature, air pollution (PM2.5), and energy consumption.
The objective is to build a Random Forest Regression model that can analyze environmental sensor data and accurately predict daily temperature, PM2.5 pollution levels, and energy usage. The model should be evaluated using performance metrics like MAE, RMSE, and R² score, and its stability should be verified using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries such as Pandas and sklearn.
2. Load dataset containing environmental sensor data.
3. Define input features (humidity, pressure, wind speed, date info) and target variables.
4. Split dataset into training and testing sets.
5. Train Random Forest Regressor model for multi-output prediction.
6. Evaluate model using MAE, RMSE, R² score and perform cross-validation.

## Program:
# Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
# Developed by: VIJIYALAKSHMI A
# RegisterNumber: 212225240185

```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample environmental dataset
data = {
    'Humidity': [60, 65, 70, 75, 80, 85, 90, 55, 50, 45],
    'Pressure': [1012, 1010, 1008, 1005, 1003, 1001, 1000, 1015, 1018, 1020],
    'WindSpeed': [5, 6, 7, 5, 4, 3, 2, 6, 7, 8],
    'Temperature': [30, 29, 28, 27, 26, 25, 24, 31, 32, 33],
    'PM2.5': [80, 85, 90, 95, 100, 110, 120, 70, 65, 60],
    'Energy': [200, 210, 220, 230, 240, 250, 260, 190, 185, 180]
}

df = pd.DataFrame(data)

# Features and Targets
X = df[['Humidity', 'Pressure', 'WindSpeed']]
y = df[['Temperature', 'PM2.5', 'Energy']]  # multi-output regression

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# -------- Evaluation Metrics --------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

# -------- Cross Validation --------
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\nCross Validation R2 Scores:", cv_scores)
print("Average CV Score:", cv_scores.mean())
```
## Output:
<img width="934" height="146" alt="Screenshot 2026-05-11 192050" src="https://github.com/user-attachments/assets/2c55e21d-0ebc-4712-8191-0c76a8041fe8" />

## Result:
The Random Forest model was successfully implemented for weather prediction. It gave good accuracy with low error and reliable results.
