# Created by Carlos Alvarez on 06-07-2025
import pandas as pd
from preprocess import preprocess_data, split_last_day_data
from postprocess import postprocess_predictions, minutes_to_hhmm
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker




if __name__ == "__main__":
    # Load the dataset from the CSV file
    dataset = pd.read_csv('dataset.csv')

    # Get the last day of data
    dataset, dataset_test_last_day = split_last_day_data(dataset)

    # Preprocess the dataset
    X, y = preprocess_data(dataset)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # First prediction model : catboost. No need to do further preprocessing.
    model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=8, loss_function='RMSE', verbose=100)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)
    # Postprocess the predictions
    predictions = postprocess_predictions(predictions)

    # Evaluate the model on the test data
    test_score = model.score(X_test, y_test)
    print(f'Test score: {test_score}')
 

    # Plot feature importances
    feature_names = X.columns
    importances = model.get_feature_importance()
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.title("CatBoost Feature Importances")
    plt.show()

    # Plot the predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predictions")
    plt.grid(True)
    plt.title("Predictions vs Actual Values")
    plt.show()

    # Check the last day of data
    X_test_last_day, y_test_last_day = preprocess_data(dataset_test_last_day)
    predictions_last_day = model.predict(X_test_last_day)
    predictions_last_day = postprocess_predictions(predictions_last_day)

    # Plot the predictions for the last day of data for each station
    stations = X_test_last_day['station_number'].unique()
    fig, axes = plt.subplots(nrows=len(stations), ncols=1, figsize=(8, 4 * len(stations)), sharex=False)

    for i, station in enumerate(stations):
        station_data = X_test_last_day[X_test_last_day['station_number'] == station]
        station_predictions = predictions_last_day[X_test_last_day['station_number'] == station]
        station_actual = y_test_last_day[X_test_last_day['station_number'] == station]

        ax = axes[i]
        ax.plot(station_data['hour_minute'], station_predictions, label='Prediction', marker='o')
        ax.plot(station_data['hour_minute'], station_actual, label='Actual', marker='x')
        ax.set_title(f'Station {station}')
        ax.set_xlabel('Hour-Minute')
        ax.set_ylabel('Number of Bikes')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(minutes_to_hhmm))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


