# Created by Carlos Alvarez
# Date: 2025-06-04

# This script fetches bike station data from the JCDecaux API and stores it in a PostgreSQL database.
# It creates a table if it does not exist and inserts the data with a timestamp.



import requests
import psycopg2
from datetime import datetime
import json

# API URL and parameters for JCDecaux API
API_URL = "https://api.jcdecaux.com/vls/v1/stations"
CONTRACT = "Seville"

def get_station_data(api_key):
    """Fetch all station data from the JCDecaux API.
    
    Returns:
        list: A list of dictionaries containing station data."""
    
    params = {
        'contract': CONTRACT,
        'apiKey': api_key
    }
    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    return response.json()


def create_table_if_not_exists(conn):
    """Create the bike_station_status table if it does not exist.
    
    Args:
        conn (psycopg2.connection): Connection to the PostgreSQL database.
    """
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS bike_station_status (
            id SERIAL PRIMARY KEY,
            station_number INT,
            timestamp TIMESTAMP,
            available_bikes INT,
            available_stands INT,
            status TEXT
        );
        """)
        conn.commit()

def insert_station_data(conn, stations):
    """Insert station data into the bike_station_status table.
    Args:
        conn (psycopg2.connection): Connection to the PostgreSQL database.
        stations (list): List of station data dictionaries.
    """
    now = datetime.now()
    with conn.cursor() as cur:
        for station in stations:
            cur.execute("""
                INSERT INTO bike_station_status (
                    station_number, timestamp,
                    available_bikes, available_stands, status
                ) VALUES (%s, %s, %s, %s, %s);
            """, (
                station['number'],
                now,
                station['available_bikes'],
                station['available_bike_stands'],
                station['status']
            ))
        conn.commit()

def main():
    print("Connecting to the database...")

    with open('db_config.json', 'r') as file:
        DB_CONFIG = json.load(file)

    with open('jcdecaux_api.json', 'r') as file:
        API_CONFIG = json.load(file)

    api_key = API_CONFIG.get('API_KEY', '')

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        print("Create the table if it doesn't exist...")
        create_table_if_not_exists(conn)

        print("Get data from the JCDecaux API...")
        stations = get_station_data(api_key)

        print(f"Inserting {len(stations)} stations data to the database...")
        insert_station_data(conn, stations)

        print("Data inserted successfully.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
