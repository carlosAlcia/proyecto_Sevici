import get_data_from_postgresql
import process_db


if __name__ == "__main__":
    print("This is the test module of the project.")
    
    # Fetch real data from the PostgreSQL database
    db_config = get_data_from_postgresql.get_db_config()
    real_data = get_data_from_postgresql.get_real_data(db_config)
    
    # Process the fetched data
    processed_data = process_db.filter_three_days(real_data)
    processed_data = process_db.fill_gaps_with_synthetic_data(processed_data)
    processed_data = process_db.split_by_day(processed_data)
    processed_data = process_db.create_more_data(processed_data,3)
    processed_data.to_csv('processed_data.csv', index=False, date_format='%d-%m-%Y-%H-%M-%S')  # Save the processed data to a CSV file
      

    # Display the first few rows of the processed data
    print("Processed Data:")
    print(processed_data.head())  # Display the first few rows of the processed data