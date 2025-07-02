# Created by Carlos Alvarez Cia on 02-07-2025
import pandas as pd
import json
import numpy as np

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


def compute_probability_factor(events_probabilities):
    """ Compute the probability factor based on the events probabilities.
    
    Args:
        events_probabilities (dict): A dictionary containing the probabilities and factors for each event.
    
    Returns:
        float: The computed probability factor.
    """
    factor = 1.0  # Start with a factor of 1 (no change)
    events = []  # List to store the events that occurred
    
    for event in events_probabilities:
        occurred = 0
        _, event_info = list(event.items())[0] 
                
        if np.random.rand() < event_info.get('probability', 0):
            factor += event_info.get('impact', 0)
            occurred = 1

        events.append(occurred)

    return factor, events


def compute_temperature_factor(temperature_data):
    temperature = int(np.random.normal(temperature_data['mean'], temperature_data['std'], 1))
    if temperature < temperature_data['min']:
        temperature = temperature_data['min']
    elif temperature > temperature_data['max']:
        temperature = temperature_data['max']

    
    if temperature < temperature_data['min_acc'] or temperature > temperature_data['max_acc']:
        return temperature_data['impact'], temperature
    else:
        return 1.0, temperature


def create_day_data(day_data, day, i, events_probabilities):

    # Create the data from the first monday of june 2025
    initial_timestamp = pd.to_datetime('2025-06-02 00:00:00')

    # Compute the temperature factor
    temperature_factor, temperature = compute_temperature_factor(events_probabilities['temperature'])

    factor, events = compute_probability_factor(events_probabilities['binary_events'])
    # Apply the temperature factor to the available stands
    factor *= temperature_factor

    day_data['timestamp'] = initial_timestamp + pd.Timedelta(days=i * 7 + day)
    # Save the size of station before applying the factor
    station_size = day_data['available_bikes'] + day_data['available_stands']
    # Apply the factor to the difference in available stands
    day_data['available_stands'] = (day_data['available_stands'] * factor).astype(int)
    # Ensure the available stands do not exceed the station size
    day_data['available_stands'] = day_data['available_stands'].clip(upper=station_size)
    # Update the available bikes accordingly
    day_data['available_bikes'] = (station_size - day_data['available_stands']).astype(int)
    for event_dict, occurred in zip(events_probabilities['binary_events'], events):
        event_name, _ = list(event_dict.items())[0]
        day_data[event_name] = occurred
    # Add temperature impact
    day_data['temperature'] = temperature
    return day_data


def create_more_data(daily_data, weeks=4):
    """ Create more data for the specified number of weeks.
    
    Args:
        daily_data (list): A list of DataFrames, each containing data for one day (thursday, friday, saturday).
        weeks (int): The number of weeks to create data for.
    
    Returns:
        pd.DataFrame: The complete data DataFrame with additional weeks of data.
    """
    complete_data = pd.DataFrame()

    workday = daily_data[0]  # Thursday data
    friday = daily_data[1]  # Friday data
    weekend = daily_data[2]  # Saturday data

    with open('create_db/probabilities.json', 'r') as file:
        events_probabilities = json.load(file)

    for i in range(weeks):
        for day in range(7):
            match day:
                # For workdays (Monday to Thursday)
                case 0 | 1 | 2 | 3 :  
                    day_data = workday.copy()
                    day_data = create_day_data(day_data, day, i, events_probabilities)
                    
                # For Friday
                case 4:
                    day_data = friday.copy()
                    day_data = create_day_data(day_data, day, i, events_probabilities)
                # For Saturday and Sunday
                case 5 | 6 :
                    day_data = weekend.copy()
                    day_data = create_day_data(day_data, day, i, events_probabilities)
            
            complete_data = pd.concat([complete_data, day_data], ignore_index=True)
    
    return complete_data