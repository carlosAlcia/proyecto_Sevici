# Created by Carlos Alvarez on 06-07-2025

import get_data_from_postgresql
import process_db


if __name__ == "__main__":

    TOTAL_MONTHS = 4
    
    # Fetch real data from the PostgreSQL database
    db_config = get_data_from_postgresql.get_db_config()
    real_data = get_data_from_postgresql.get_real_data(db_config)
    
    # Process the fetched data
    processed_data = process_db.filter_three_days(real_data)
    processed_data = process_db.fill_gaps_with_synthetic_data(processed_data)
    processed_data = process_db.split_by_day(processed_data)
    processed_data = process_db.create_more_data(processed_data,4*TOTAL_MONTHS)
    processed_data.to_csv('../prediction/dataset.csv', index=False, date_format='%d-%m-%Y-%H-%M-%S')  # Save the processed data to a CSV file
      