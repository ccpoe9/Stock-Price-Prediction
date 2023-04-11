import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(directory):
    # Initialize an empty list to store stock data from multiple files
    stock_data = []

    # Get a list of all the files with a ".txt" extension in the directory
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]

    for filename in tqdm(files, desc="Loading data"):
        file_path = os.path.join(directory, filename)
        # Check if the file is not empty
        if os.stat(file_path).st_size != 0:
            # Extract the stock symbol from the file
            symbol = filename.split(".")[0]
            # Read the stock data from the file and store it in a DataFrame
            stock_df = pd.read_csv(file_path, parse_dates=["Date"])
            # Add a new column to the DataFrame containing the stock symbol
            stock_df["Symbol"] = symbol
            # Append the stock DataFrame to the stock_data list
            stock_data.append(stock_df)

    # Combine all the stock DataFrames in the list into a single DataFrame
    combined_df = pd.concat(stock_data)
    return combined_df


def preprocess_data(data, days_ahead=30):
    # Create a new column 'Label' that is True when
    # the close price is higher in 'days_ahead' days
    data["Label"] = data["Close"].shift(-days_ahead) > data["Close"]
    # Convert the 'Label' column to an integer (True = 1, False = 0)
    data["Label"] = data["Label"].astype(int)
    # Add a new column with the 30-day moving average of the close price
    data["30_day_moving_average"] = data["Close"].rolling(window=30).mean()
    # Convert the 'Date' column to ordinal numbers
    data["Date"] = data["Date"].apply(lambda x: x.toordinal())

    # Drop rows with NaN values caused by the moving average calculation
    data.dropna(inplace=True)
    return data


def train_test(data):
    X = data.drop(columns="Label")
    y = data["Label"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create the logistic regression model
    lr = LogisticRegression(max_iter=1000)

    # Perform cross-validation
    scores = cross_val_score(lr, X_train, y_train, cv=5)
    print("Cross-validation scores:", scores)
    print("Average cross-validation score:", scores.mean())

    # Train the model on the entire training set
    lr.fit(X_train, y_train)

    # Test the model on the testing set
    y_pred = lr.predict(X_test)

    # Evaluate the performance of the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))


def main():
    # Load and preprocess the data
    data = load_data("data/Stocks/")
    data = preprocess_data(data)

    # Use LabelEncoder to convert stock symbols to numerical format
    # So the model can distinguish between different stocks
    le = LabelEncoder()
    data["Symbol"] = le.fit_transform(data["Symbol"])

    train_test(data)


if __name__ == "__main__":
    main()
