import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def preprocess_data(df, days_ahead=30, window_size=30):
    # Create a new column 'Label' to hold the
    # close price 'days_ahead' days in the future
    df["Label"] = df["Close"].shift(-days_ahead)
    # Add a new column with the 30-day moving average of the close price
    df["30_day_moving_average"] = df["Close"].rolling(window=window_size).mean()

    # Drop rows with NaN values caused by the moving average calculation
    df.dropna(inplace=True)
    return df


def load_data(stock):
    stock = stock.upper()
    df = pd.read_csv(
        f"nasdaq/{stock}.csv",
        parse_dates=["Date"],
        dayfirst=True,
    )
    df = preprocess_data(df)
    df.index = df.pop("Date")
    return df


def split_data(X, y, test_ratio=0.1, val_ratio=0.1):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_ratio + val_ratio, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio / (test_ratio + val_ratio), shuffle=False
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_test(X, y):
    # Split the data into training and testing sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Create the logistic regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Make predictions
    train_pred = lr.predict(X_train).flatten()
    val_pred = lr.predict(X_val).flatten()
    test_pred = lr.predict(X_test).flatten()

    # Evaluate the performance of the model
    trainScore = np.sqrt(mean_squared_error(y_train, train_pred))
    print("Train Score: %.2f RMSE" % (trainScore))
    valScore = np.sqrt(mean_squared_error(y_val, val_pred))
    print("Cross Validation Score: %.2f RMSE" % (valScore))
    testScore = np.sqrt(mean_squared_error(y_test, test_pred))
    print("Test Score: %.2f RMSE" % (testScore))

    return train_pred, val_pred, test_pred


def main():
    # Load and preprocess the data
    df = load_data("AAL")

    # Split the data into features and target
    X = df.drop(columns="Label")
    y = df["Label"]

    # Train and test the model
    train_pred, val_pred, test_pred = train_test(X, y)

    sorted_dates = df.index.sort_values()
    sorted_close_prices = df.loc[sorted_dates, "Close"]

    dates_train = sorted_dates[: len(train_pred)]
    dates_val = sorted_dates[len(train_pred) : len(train_pred) + len(val_pred)]
    dates_test = sorted_dates[len(train_pred) + len(val_pred) :]

    # Plot the close prices
    plt.plot(sorted_dates, sorted_close_prices, label="Close Prices", alpha=0.7)
    # Plot the predictions
    plt.plot(dates_train, train_pred, label="Training Predictions")
    plt.plot(dates_val, val_pred, label="Validation Predictions")
    plt.plot(dates_test, test_pred, label="Testing Predictions")

    # Add labels and legend
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Close Price and Predictions")
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
