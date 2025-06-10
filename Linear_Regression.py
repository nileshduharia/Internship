import numpy as np
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load and prepare the data
df = pd.read_csv('Desktop/Internship/ML Model/cost.csv')
print(df)

# Drop the 'Location' column as it's categorical and we're not handling it in this simple model
df = df.drop(columns=['Location'])
print(df)

# Split into features (x) and target (y)
x = df.iloc[:, :5]
y = df.iloc[:, -1]

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Scale the features
scaler = StandardScaler()
x_tscale = scaler.fit_transform(x_train)
x_testscale = scaler.fit_transform(x_test)

# Create and train the linear regression model
clf = LinearRegression()
clf.fit(x_tscale, y_train)

# Make predictions
y_predict = clf.predict(x_testscale)
print(y_predict)

# Print the actual test values for comparison
print(y_test)

# Plot actual vs predicted values
plt.scatter(y_test, y_predict, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # perfect line
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()