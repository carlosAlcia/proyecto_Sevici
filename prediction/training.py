# Created by Carlos Alvarez on 06-07-2025
import pandas as pd
from preprocess import preprocess_data
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Load the dataset from the CSV file
    dataset = pd.read_csv('dataset.csv')

    # TODO: To have the last day of data, we need to get it here, before the preprocess.

    # Preprocess the dataset
    dataset = preprocess_data(dataset)

    # First prediction model : catboost. No need to do further preprocessing.
    model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6, loss_function='RMSE', verbose=100)
    X = dataset.drop(columns=['available_bikes'])
    y = dataset['available_bikes']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

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





