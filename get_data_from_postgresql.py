# Created by Carlos Alvarez Cia on 02-07-2025

import pandas as pd
from sqlalchemy import create_engine
import json

def get_real_data(db_config):
    """ Fetch the real data from the PostgreSQL database. Filter for station numbers 88 and 247.
    
    Returns:
        """
    
    engine = create_engine(f"postgresql+psycopg2://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}")
    query = "SELECT station_number, timestamp, available_bikes, available_stands FROM bike_station_status WHERE station_number IN (88, 247);"
    real_data = pd.read_sql(query, engine)

    return real_data

def get_db_config():
    """ Get the database configuration from a file.
    
    Returns:
        dict: Database configuration.
    """
    
    db_config = {}
    with open('db_config.json', 'r') as file:
        db_config = json.load(file)
    return db_config


