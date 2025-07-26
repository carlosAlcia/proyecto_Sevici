# Created by Carlos Alvarez on 06-07-2025
import pandas as pd
from preprocess import preprocess_data, split_last_day_data
from postprocess import postprocess_predictions
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from plots_results import plot_last_day, plot_feature_importances, plot_predictions_vs_actual




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
    plot_feature_importances(model, X.columns)

    # Plot the predictions vs actual values
    plot_predictions_vs_actual(y_test, predictions)

    # Check the last day of data
    X_test_last_day, y_test_last_day = preprocess_data(dataset_test_last_day)
    predictions_last_day = model.predict(X_test_last_day)
    predictions_last_day = postprocess_predictions(predictions_last_day)

    # Plot the predictions for the last day of data
    plot_last_day(X_test_last_day, predictions_last_day, y_test_last_day)




