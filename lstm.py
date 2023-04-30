import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers


def preprocess_data(df, days_ahead=30, window_size=30):
    df["Label"] = df["Close"].shift(-days_ahead) > df["Close"]
    df["30_day_moving_average"] = df["Close"].rolling(window=window_size).mean()

    df.dropna(inplace=True)
    return df


def df_to_windowed_df(df, n=3):
    last_date = df.index.max()
    target_date = df.index[n]

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = df.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            print(f"Error: Window of size {n} is too large for date {target_date}")
            return

        values = df_subset["Close"].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = df.loc[target_date : target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split("T")[0]
        year_month_day = next_date_str.split("-")
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df["Target Date"] = dates

    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f"Target-{n-i}"] = X[:, i]

    ret_df["Target"] = Y

    return ret_df


def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()

    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)


if __name__ == "__main__":
    df = pd.read_csv("nasdaq/AAL.csv", parse_dates=["Date"], dayfirst=True)
    df = preprocess_data(df)

    df.index = df.pop("Date")

    X = df.drop(columns="Label")
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    windowed_df = df_to_windowed_df(df, n=3)
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    q_80 = int(len(dates) * 0.8)
    q_90 = int(len(dates) * 0.9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
    model = Sequential(
        [
            layers.Input((3, 1)),
            layers.LSTM(64),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["mean_absolute_error"]
    )

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

    train_pred = model.predict(X_train).flatten()
    val_pred = model.predict(X_val).flatten()
    test_pred = model.predict(X_test).flatten()

    # training data
    plt.plot(dates_train, train_pred)
    plt.plot(dates_train, y_train)

    # validation data
    plt.plot(dates_val, val_pred)
    plt.plot(dates_val, y_val)

    # testing data
    plt.plot(dates_test, test_pred)
    plt.plot(dates_test, y_test)
    plt.legend(
        [
            "Training Predictions",
            "Training Observations",
            "Validation Predictions",
            "Validation Observations",
            "Testing Predictions",
            "Testing Observations",
        ]
    )
    plt.show()

    trainScore = np.sqrt(mean_squared_error(y_train, train_pred))
    print("Train Score: %.2f RMSE" % (trainScore))
    valScore = np.sqrt(mean_squared_error(y_val, val_pred))
    print("Cross Validation Score: %.2f RMSE" % (trainScore))
    testScore = np.sqrt(mean_squared_error(y_test, test_pred))
    print("Test Score: %.2f RMSE" % (testScore))
