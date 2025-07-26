import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from postprocess import minutes_to_hhmm

def plot_last_day(X_test_last_day: pd.DataFrame, predictions_last_day: pd.Series, y_test_last_day: pd.Series):
    # Plot the predictions for the last day of data for each station
    stations = X_test_last_day['station_number'].unique()
    _, axes = plt.subplots(nrows=len(stations), ncols=1, figsize=(8, 4 * len(stations)), sharex=False)

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

def plot_feature_importances(model, feature_names:list):
    # Plot feature importances
    importances = model.get_feature_importance()
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.title("CatBoost Feature Importances")
    plt.show()


def plot_predictions_vs_actual(y_test: pd.Series, predictions: pd.Series):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predictions")
    plt.grid(True)
    plt.title("Predictions vs Actual Values")
    plt.show()