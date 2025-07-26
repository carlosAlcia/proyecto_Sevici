# Created by Carlos Alvarez on 06-07-2025
import pandas as pd
from preprocess import preprocess_data, split_last_day_data, normalize_data
from postprocess import postprocess_predictions
from plots_results import plot_last_day, plot_predictions_vs_actual
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# For the NN model
from model_nn import ModelNN
import torch



if __name__ == "__main__":
    # Load the dataset from the CSV file
    dataset = pd.read_csv('dataset.csv')

    # Get the last day of data
    dataset, dataset_test_last_day = split_last_day_data(dataset)

    # Preprocess the dataset
    X, y = preprocess_data(dataset)
    # Split the dataset into training, validation and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    # Scale the values
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_feat, scaler_target = \
        normalize_data(X_train, X_val, X_test, y_train, y_val, y_test)

    model = ModelNN(input_size=X_train.shape[1], hidden_size=[256, 128], output_size=1)
        
    # Fit the model
    model,train_losses, val_losses = model.fit(X_train, y_train, X_val, y_val, epochs=2000, lr=0.005, patience=100, plot_graphs=True)

    # Make predictions
    predictions_nn = model.predict(X_test).numpy()
    # Postprocess the predictions
    predictions_nn = scaler_target.inverse_transform(predictions_nn).flatten()
    predictions_nn = postprocess_predictions(predictions_nn)
    # Evaluate the model on the test data
    test_score_nn = model.score(X_test, torch.tensor(y_test, dtype=torch.float32).view(-1, 1))
    print(f'Test score (NN): {test_score_nn}')

    y_test = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
    # Plot the predictions vs actual values for the NN model
    plot_predictions_vs_actual(y_test, predictions_nn)

    # Check the last day of data
    X_test_last_day, y_test_last_day = preprocess_data(dataset_test_last_day)
    X_test_last_day_values = scaler_feat.transform(X_test_last_day)
    predictions_last_day = model.predict(X_test_last_day_values).numpy()
    predictions_last_day = scaler_target.inverse_transform(predictions_last_day).flatten()
    predictions_last_day = postprocess_predictions(predictions_last_day)

    # Plot the predictions for the last day of data for each station
    plot_last_day(X_test_last_day, predictions_last_day, y_test_last_day)





