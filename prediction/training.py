# Created by Carlos Alvarez on 06-07-2025
import pandas as pd
from preprocess import preprocess_data



if __name__ == "__main__":
    # Load the dataset from the CSV file
    dataset = pd.read_csv('dataset.csv')

    # Preprocess the dataset
    dataset = preprocess_data(dataset)

    print("Preprocessed dataset:")
    print(dataset.head())
