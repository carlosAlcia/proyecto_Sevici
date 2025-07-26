import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def split_last_day_data(dataset : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Split the dataset to get the last day of data and the training dataset.
    
    Args:
        dataset (pd.DataFrame): The dataset to split.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training dataset and the last day dataset.
    """

    # Ensure the 'timestamp' column is in datetime format
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], format='%d-%m-%Y-%H-%M-%S', errors='raise')
    
    # Get the last day of data
    last_day = dataset['timestamp'].dt.day.max()
    dataset_test_last_day = dataset[dataset['timestamp'].dt.day == last_day]

    # Filter the dataset to only include data before the last day
    # This is to avoid data leakage in the training phase.
    dataset_train = dataset[dataset['timestamp'].dt.day < last_day]

    return dataset_train, dataset_test_last_day


def preprocess_data(dataset : pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """ Preprocess the dataset.
    
    Args:
        dataset (pd.DataFrame): The dataset to preprocess.
    
    Returns:
        tuple[pd.DataFrame, pd.Series]: (X, y) where X is the features and y is the target variable.
        """

    # Ensure the 'timestamp' column is in datetime format
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], format='%d-%m-%Y-%H-%M-%S', errors='raise')

    # Create additional features from the 'timestamp' column
    # Extract the day of the week from the 'timestamp' column
    dataset['day_of_week'] = dataset['timestamp'].dt.dayofweek

    # Extract the hour and minute from the 'timestamp' column
    dataset['hour'] = dataset['timestamp'].dt.hour
    dataset['minute'] = dataset['timestamp'].dt.minute
    dataset['hour_minute'] = dataset['hour'] * 60 + dataset['minute']

    # Drop the 'timestamp' column 
    if 'timestamp' in dataset.columns:
        dataset.drop(columns=['timestamp'], inplace=True)

    # Drop the 'available_stands' column 
    if 'available_stands' in dataset.columns:
        dataset.drop(columns=['available_stands'], inplace=True)

    dataset.reset_index(drop=True)

    X = dataset.drop(columns=['available_bikes'])
    y = dataset['available_bikes']

    return X, y


def normalize_data(X:pd.DataFrame, y:pd.Series, X_scaler:MinMaxScaler=None, y_scaler:MinMaxScaler=None) -> tuple[pd.DataFrame, pd.Series]:
    """ Normalize the data using Min-Max scaling.
    
    Args:
        X (pd.DataFrame): Features to normalize.
        y (pd.Series): Target variable to normalize.
        X_scaler (MinMaxScaler, optional): Scaler for features. If None, a new scaler will be created.
        y_scaler (MinMaxScaler, optional): Scaler for target variable. If None, a new scaler will be created.
    Returns:
        tuple[pd.DataFrame, pd.Series, MinMaxScaler]: Normalized features, normalized target variable, and the scaler used.
    """

    if X_scaler is None:
        X_scaler = MinMaxScaler()
        X_scaler.fit(X)
    if y_scaler is None:
        y_scaler = MinMaxScaler()
        y_scaler.fit(y.values.reshape(-1, 1))

    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(y.values.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled, X_scaler, y_scaler
    