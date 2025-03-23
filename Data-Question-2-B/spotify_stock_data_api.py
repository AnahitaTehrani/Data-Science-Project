import requests
import pandas as pd
from datetime import datetime
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create output directory for the CSV files
output_dir = os.path.join(script_dir, "spotify_stock_data_api")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# API configuration
API_KEY = "cd72ff0b5ead5fbeb7ff86ac4140e2e8"
BASE_URL = "http://api.marketstack.com/v1/eod"
SYMBOL = "SPOT"  # Spotify stock symbol
DATE_FROM = "2024-03-01"
DATE_TO = "2024-03-24"

# Function to fetch all data with pagination
def fetch_stock_data():
    all_data = []
    offset = 0
    limit = 100  # Maximum allowed by API
    
    while True:
        # Set up parameters for API request
        params = {
            'access_key': API_KEY,
            'symbols': SYMBOL,
            'date_from': DATE_FROM,
            'date_to': DATE_TO,
            'limit': limit,
            'offset': offset
        }
        
        # Make API request
        response = requests.get(BASE_URL, params=params)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            break
        
        # Parse JSON response
        data = response.json()
        
        # Add data to our list
        if 'data' in data and data['data']:
            all_data.extend(data['data'])
            
            # Check if we've reached the end of the data
            if len(data['data']) < limit:
                break
                
            # Update offset for next request
            offset += limit
        else:
            break
    
    return all_data

# Main execution
if __name__ == "__main__":
    print("Fetching Spotify stock data...")
    stock_data = fetch_stock_data()
    
    if stock_data:
        # Convert to DataFrame
        df = pd.DataFrame(stock_data)
        
        # Format the date column
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Sort by date (newest first)
        df = df.sort_values('date', ascending=False)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"spotify_stock_data_{timestamp}.csv"
        df.to_csv(os.path.join(output_dir, csv_filename), index=False)
        
        print(f"Successfully saved {len(df)} records to {csv_filename}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    else:
        print("No data retrieved or error occurred.")
