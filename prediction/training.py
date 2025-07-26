# Created by Carlos Alvarez on 06-07-2025
import pandas as pd
import numpy as np
from preprocess import preprocess_data, split_last_day_data, normalize_data
from postprocess import postprocess_predictions
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from plots_results import plot_last_day, plot_feature_importances, plot_predictions_vs_actual
from model_nn import ModelNN


# Definitions
MODEL_TYPE = 'nn'  # Change to 'nn' for neural network model
PARAMS_MODEL_CATBOOST = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 8,
    'loss_function': 'RMSE',
    'verbose': 100
}
PARAMS_MODEL_NN = {
    'hidden_size': [64, 64, 32, 32],
    'output_size': 1,
    'dropout_rate': 0
}

PARAMS_TRAINING_NN = {
    'epochs': 2000,
    'lr': 0.005,
    'early_stopping': True,
    'patience': 100,
    'plot_graphs': True
}


class ModelML:
    """A wrapper class for different machine learning models."""
    def __init__(self, model_type:str='catboost', **kwargs):
        """Initialize the model."""
        self.model_type = model_type
        # Initialize the model based on the type
        if model_type == 'catboost':
            self.model = CatBoostRegressor(**kwargs)
        elif model_type == 'nn':
            self.model = ModelNN(**kwargs)
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        
        self.X_scaler = None
        self.y_scaler = None


    def fit(self, X_train:pd.DataFrame, y_train:pd.Series, X_val:pd.DataFrame, y_val:pd.Series, **kwargs):
        """Fit the model to the training data."""
        if self.model_type == 'nn':
            # Normalize the data for neural network
            X_train, y_train, self.X_scaler, self.y_scaler = normalize_data(X_train, y_train)
            X_val, y_val, _, _ = normalize_data(X_val, y_val, X_scaler=self.X_scaler, y_scaler=self.y_scaler)
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), **kwargs)

    def predict(self, X:pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model_type == 'nn':
            X = self.X_scaler.transform(X)
        predictions = self.model.predict(X)
        if self.model_type == 'nn':
            predictions = self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        return postprocess_predictions(predictions)

    def score(self, X:pd.DataFrame, y:pd.Series, print_score:bool=False) -> float:
        """Evaluate the model on the test data."""
        if self.model_type == 'nn':
            X = self.X_scaler.transform(X)
            y = self.y_scaler.transform(y.values.reshape(-1, 1)).flatten()
        test_score = self.model.score(X, y)
        if print_score:
            print(f'Test score: {test_score}')
        return test_score





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

    # Load and train the model selected by MODEL_TYPE
    if MODEL_TYPE == 'catboost':
        model = ModelML(MODEL_TYPE, **PARAMS_MODEL_CATBOOST)
        model.fit(X_train, y_train, X_val, y_val)
    elif MODEL_TYPE == 'nn':
        model = ModelML(MODEL_TYPE, input_size=X_train.shape[1], **PARAMS_MODEL_NN)
        model.fit(X_train, y_train, X_val, y_val, **PARAMS_TRAINING_NN)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Evaluate the model on the test data
    test_score = model.score(X_test, y_test, print_score=True)

    if MODEL_TYPE == 'catboost':
        # Plot feature importances
        plot_feature_importances(model.model, X.columns)
    
    # Plot the predictions vs actual values
    plot_predictions_vs_actual(y_test, predictions)

    # Check the last day of data
    X_test_last_day, y_test_last_day = preprocess_data(dataset_test_last_day)
    predictions_last_day = model.predict(X_test_last_day)
    predictions_last_day = postprocess_predictions(predictions_last_day)

    # Plot the predictions for the last day of data
    plot_last_day(X_test_last_day, predictions_last_day, y_test_last_day)




