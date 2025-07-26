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


def normalize_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, MinMaxScaler, MinMaxScaler]:
    """ Normalize the data using Min-Max scaling.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.Series): Training target variable.
        y_val (pd.Series): Validation target variable.
        y_test (pd.Series): Test target variable.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: Normalized features and target variables.
    """

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler_X, scaler_y