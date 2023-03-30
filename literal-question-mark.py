import os
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

stocks_path = "data/Stocks/"


def preprocess_data(df):
    # Calculate the percentage change in closing price
    df["Close_pct_change"] = df["Close"].pct_change()

    # Calculate the label (stock direction)
    df["Label"] = np.where(
        df["Close_pct_change"].shift(-30) > 0, 1, 0
    )  # 30 trading days for the next month

    # Define the past 30 days window for input features
    window = 30

    # Create new columns for past n days of closing prices as input features
    for i in range(1, window + 1):
        df[f"Close_lag_{i}"] = df["Close"].shift(i)

    # Drop the first (window + 30) rows due to lack of historical data and potential future leakage
    df = df.dropna()

    # Separate the input features (X) and the target label (y)
    X = df[[f"Close_lag_{i}" for i in range(1, window + 1)]]
    y = df["Label"]

    return X, y


def train_model_incrementally(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.txt"))

    log_reg = SGDClassifier(loss="log_loss", random_state=42)

    for file in all_files:
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")
            continue

        X, y = preprocess_data(df)

        if X.empty or y.empty or len(np.unique(y)) < 2:
            print(f"Skipping file with insufficient data: {file}")
            continue

        # Scale the input features using MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the logistic regression model incrementally
        log_reg.partial_fit(X_scaled, y, classes=np.unique(y))

    return log_reg


if __name__ == "__main__":
    # Train the logistic regression model incrementally
    log_reg = train_model_incrementally(stocks_path)

    # Test the model on a specific stock
    test_stock_data = pd.read_csv("data/Stocks/tsla.us.txt")
    X_test, y_test = preprocess_data(test_stock_data)

    # Scale the input features using MinMaxScaler
    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    # Make predictions on the test set
    y_pred = log_reg.predict(X_test_scaled)

    # Calculate the accuracy score and confusion matrix
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
