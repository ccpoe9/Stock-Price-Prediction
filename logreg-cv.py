import numpy as np
import pandas as pd
import os
import glob
import random
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Set the path to the stocks folder
stocks_path = "data/Stocks/"


def concatenate_files(folder_path, sample_size=None):
    all_files = glob.glob(os.path.join(folder_path, "*.txt"))

    if sample_size is not None:
        all_files = random.sample(all_files, sample_size)

    data_list = []

    for file in tqdm(all_files, desc="Processing stock files"):
        # grab ticker symbol by extracting the file name
        ticker = os.path.basename(file).split(".")[0]
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")
            continue

        df["Ticker"] = ticker
        data_list.append(df)

    # Return the list of DataFrames
    return data_list


def cross_val(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold = 1
    for train_index, val_index in tscv.split(X, y):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)
        fold_acc = accuracy_score(y_val_fold, y_pred_fold)

        print(f"Fold {fold} accuracy: {fold_acc:.4f}")
        fold += 1


def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    # Read and concatenate a sample of stock files
    sample_size = 1000
    all_stocks_data_list = concatenate_files(stocks_path, sample_size=sample_size)

    # Initialize empty lists for the input features and labels
    X_list = []
    Y_list = []

    for stock_data in all_stocks_data_list:
        # Calculate the percentage change in closing price
        stock_data["Close_pct_change"] = stock_data["Close"].pct_change()

        # Calculate the label (stock direction)
        stock_data["Label"] = np.where(
            stock_data["Close_pct_change"].shift(-30) > 0, 1, 0
        )  # 30 trading days for the next month

        # Define the past 30 days window for input features
        window = 30

        # Create new columns for past n days of closing prices as input features
        for i in range(1, window + 1):
            stock_data[f"Close_lag_{i}"] = stock_data["Close"].shift(i)

        # Drop the first (window + 30) rows for each stock due to lack of historical data and potential future leakage
        stock_data = stock_data.dropna().reset_index(drop=True)

        # Append the processed stock data to the lists
        X_list.append(stock_data[[f"Close_lag_{i}" for i in range(1, window + 1)]])
        Y_list.append(stock_data["Label"])

    # Concatenate the input features and labels for all stocks
    X = pd.concat(X_list, axis=0, ignore_index=True)
    Y = pd.concat(Y_list, axis=0, ignore_index=True)

    # Scale the input features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and testing sets using stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # Perform cross-validation using logistic regression
    print("Cross-validation using logistic regression:")
    log_reg = LogisticRegression(random_state=42)
    cross_val(log_reg, X_train, y_train, 50)

    # Train the logistic regression model on the full training set and evaluate it
    print("\nEvaluating logistic regression model on the test set:")
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    evaluate_model(y_test, y_pred)
