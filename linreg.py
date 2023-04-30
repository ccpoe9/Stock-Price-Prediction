import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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


def train_test(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the logistic regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Make predictions
    train_pred = lr.predict(X_train).flatten()
    test_pred = lr.predict(X_test).flatten()

    # Evaluate the performance of the model
    trainScore = np.sqrt(mean_squared_error(y_train, train_pred))
    print("Train Score: %.2f RMSE" % (trainScore))
    testScore = np.sqrt(mean_squared_error(y_test, test_pred))
    print("Test Score: %.2f RMSE" % (testScore))


def main():
    # Load and preprocess the data
    df = load_data("AAL")

    # Split the data into features and target
    X = df.drop(columns="Label")
    y = df["Label"]

    # Train and test the model
    train_test(X, y)


if __name__ == "__main__":
    main()
