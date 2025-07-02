# Created by Carlos Alvarez Cia on 02-07-2025
import get_data_from_postgresql as data_fetcher

if __name__ == "__main__":
    print("This is the main module of the project.")
    db_config = data_fetcher.get_db_config()
    real_data = data_fetcher.get_real_data(db_config)
    print("Data fetched successfully.")
    print(real_data.head())  # Display the first few rows of the fetched data