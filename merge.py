import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Set the path to the stocks and ETFs folders
stocks_path = "data/Stocks/"
etfs_path = "data/ETFs/"


# Create a function to read and concatenate all files in a folder
def concatenate_files(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.txt"))
    data_list = []

    for file in all_files:
        ticker = os.path.basename(file).split(".")[0]
        try:
            df = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")
            continue

        df["Ticker"] = ticker
        data_list.append(df)

    return pd.concat(data_list, axis=0, ignore_index=True)


# Read and concatenate all stock and ETF files
stocks_data = concatenate_files(stocks_path)
etfs_data = concatenate_files(etfs_path)

# Combine the stocks and ETFs DataFrames
all_data = pd.concat([stocks_data, etfs_data], axis=0, ignore_index=True)

# Proceed with the preprocessing and modeling steps as before, but now using the 'all_data' DataFrame

# Calculate the percentage change in closing price
all_data["Close_pct_change"] = all_data.groupby("Ticker")["Close"].pct_change()

# Calculate the label (stock direction)
all_data["Label"] = np.where(
    all_data["Close_pct_change"].shift(-20) > 0, 1, 0
)  # 20 trading days for the next month

# Create new columns for past n days of closing prices as input features
for i in range(1, window + 1):
    all_data[f"Close_lag_{i}"] = all_data.groupby("Ticker")["Close"].shift(i)

# Drop the first (window + 20) rows for each stock due to lack of historical data and potential future leakage
all_data = all_data.groupby("Ticker").apply(lambda x: x.dropna()).reset_index(drop=True)

# Separate the input features (X) and the target label (y)
X = all_data[[f"Close_lag_{i}" for i in range(1, window + 1)]]
y = all_data["Label"]

# Scale the input features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
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
