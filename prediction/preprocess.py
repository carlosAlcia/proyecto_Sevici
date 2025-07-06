import pandas as pd

def preprocess_data(dataset):
    """ Preprocess the dataset.
    
    Args:
        dataset (pd.DataFrame): The dataset to preprocess.
    
    Returns:
        pd.DataFrame: The preprocessed dataset.
    """

    # Ensure the 'timestamp' column is in datetime format
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], format='%d-%m-%Y-%H-%M-%S', errors='raise')

    # Create additional features from the 'timestamp' column
    # Extract the day of the week from the 'timestamp' column
    dataset['day_of_week'] = dataset['timestamp'].dt.dayofweek

    # Extract the hour and minute from the 'timestamp' column
    dataset['hour'] = dataset['timestamp'].dt.hour
    dataset['minute'] = dataset['timestamp'].dt.minute

    # Drop the 'timestamp' column 
    if 'timestamp' in dataset.columns:
        dataset.drop(columns=['timestamp'], inplace=True)

    # Drop the 'available_stands' column 
    if 'available_stands' in dataset.columns:
        dataset.drop(columns=['available_stands'], inplace=True)
    
    return dataset.reset_index(drop=True)

