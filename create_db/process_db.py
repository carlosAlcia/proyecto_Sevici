# Created by Carlos Alvarez Cia on 02-07-2025
import pandas as pd

def filter_three_days(real_data):
    """ Get the first three days of data from the real data. Filter from 05-06-25 to 08-06-25.
    
    Args:
        real_data (pd.DataFrame): The real data DataFrame.
    
    Returns:
        pd.DataFrame: The first full three days of data.
    """
    real_data['timestamp'] = pd.to_datetime(real_data['timestamp'])
    start_date = pd.to_datetime('2025-06-05')
    end_date = pd.to_datetime('2025-06-08')
  
    return real_data[(real_data['timestamp'] >= start_date) & (real_data['timestamp'] <= end_date)]



def fill_gaps_with_synthetic_data(real_data):
    """ Fill gaps in the real data with synthetic data. If the timestamps differ more than 15 minutes, fill the gap
    with previous values.

    Args:
        real_data (pd.DataFrame): The real data DataFrame.
    Returns:
        pd.DataFrame: The filled data DataFrame.
    """
    # Create a new DataFrame to hold the filled data
    filled_data_df = pd.DataFrame(columns=real_data.columns)

    # Fill the gaps for each station
    for station in pd.unique(real_data['station_number']):
        # Filter the data for the current station
        station_data = real_data[real_data['station_number'] == station].copy()

        # Ensure the timestamp column is in datetime format
        station_data['timestamp'] = pd.to_datetime(station_data['timestamp'])
        # Sort the data by timestamp
        station_data = station_data.sort_values(by='timestamp')
        # Set the timestamp as the index
        station_data.set_index('timestamp', inplace=True)
        # Resample the data to fill gaps with 15-minute intervals
        filled_data_station = station_data.resample('15min').ffill().reset_index()
        # Fill any remaining NaN values with the last valid observation
        filled_data_station.ffill(inplace=True)
        # Add last line for 23:45 of the last day as a copy of the last row
        last_timestamp = filled_data_station['timestamp'].max()
        last_row = filled_data_station.iloc[-1].copy()
        last_row['timestamp'] = last_timestamp + pd.Timedelta(minutes=15)
        filled_data_station = pd.concat([filled_data_station, pd.DataFrame([last_row])], ignore_index=True)
        # Modify the first row, it has no previous value to fill
        first_timestamp = filled_data_station['timestamp'].min()
        first_row = filled_data_station.iloc[1].copy()
        first_row['timestamp'] = first_timestamp
        filled_data_station.iloc[0] = first_row
        # Reset the index to have a clean DataFrame
        filled_data_station.reset_index(drop=True, inplace=True)
        # Add the station number back to the filled data
        filled_data_station['station_number'] = station
        # Append the filled data back to the original DataFrame
        filled_data_df = pd.concat([filled_data_df, filled_data_station], ignore_index=True)

    return filled_data_df

def split_by_day(data):
    """ Split the data by day.
    
    Args:
        data (pd.DataFrame): The data DataFrame.
    
    Returns:
        list: A list of DataFrames, each containing data for one day. (Thursday, Friday, Saturday)
    """
    data['date'] = data['timestamp'].dt.date
    return [group for _, group in data.groupby('date')]