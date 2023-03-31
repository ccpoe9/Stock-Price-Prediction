import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv("data/Stocks/tsla.us.txt")

# Calculate the percentage change in closing price
data["Close_pct_change"] = data["Close"].pct_change()

# Calculate the label (stock direction)
data["Label"] = np.where(
    data["Close_pct_change"].shift(-30) > 0, 1, 0
)  # 30 trading days for the next month

# Define the past 30 days window for input features
window = 30

# Create new columns for past n days of closing prices as input features
for i in range(1, window + 1):
    data[f"Close_lag_{i}"] = data["Close"].shift(i)

# Drop the first (window + 30) rows due to lack of historical data and potential future leakage
data = data.dropna()

print(data[["Close", "Close_pct_change", "Label"]])

# Separate the input features (X) and the target label (y)
X = data[[f"Close_lag_{i}" for i in range(1, window + 1)]]
Y = data["Label"]

# Scale the input features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# Create and train the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)

# Calculate the accuracy score and confusion matrix
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)
