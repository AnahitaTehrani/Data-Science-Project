import requests
import pandas as pd
import os
import datetime

api_key = ''

# If no API key is found, prompt the user
if not api_key:
    api_key = input("Please enter your Finnhub API key: ")

# Finnhub API endpoint for company news
url = "https://finnhub.io/api/v1/company-news"

# Define parameters
symbol = "SPOT"  # Spotify stock symbol

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create output directory for the CSV files
output_dir = os.path.join(script_dir, "spotify_news_2024_2025_finnhub")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to fetch news data
def fetch_news_data(symbol, start_date, end_date, api_key):
    params = {
        'symbol': symbol,
        'from': start_date,
        'to': end_date,
        'token': api_key
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Function to get date ranges for months in 2024 and 2025
def get_monthly_date_ranges():
    date_ranges = []
    
    # January and February 2025
    for month in range(1, 3):
        start_date = datetime.date(2025, month, 1)
        if month < 12:
            end_date = datetime.date(2025, month + 1, 1) - datetime.timedelta(days=1)
        else:
            end_date = datetime.date(2025, 12, 31)
        
        date_ranges.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 2025))
    
    # March to December 2024
    # for month in range(3, 13):
    #     start_date = datetime.date(2024, month, 1)
    #     if month < 12:
    #         end_date = datetime.date(2024, month + 1, 1) - datetime.timedelta(days=1)
    #     else:
    #         end_date = datetime.date(2024, 12, 31)
        
    #     date_ranges.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 2024))
    
    return date_ranges

# Main function
def main():
    date_ranges = get_monthly_date_ranges()
    
    for month_info in date_ranges:
        start_date, end_date, year = month_info
        # Extract month number from the start_date
        month_num = int(start_date.split('-')[1])
        month_name = datetime.date(year, month_num, 1).strftime('%B')
        
        print(f"\nFetching Spotify news data for {month_name} {year} ({start_date} to {end_date})...")
        
        news_data = fetch_news_data(symbol, start_date, end_date, api_key)
        
        if not news_data:
            print(f"No data retrieved or an error occurred for {month_name} {year}.")
            continue
        
        print(f"Retrieved {len(news_data)} news articles for {month_name} {year}.")
        
        # Convert to DataFrame
        df = pd.DataFrame(news_data)
        
        # Check if we received any data
        if df.empty:
            print(f"No news data available for {month_name} {year}.")
            continue
        
        # Add timestamp column converted from unix time
        if 'datetime' in df.columns:
            # Convert Unix timestamp to datetime then format as YYYY-MM-DD
            df['date'] = pd.to_datetime(df['datetime'], unit='s').dt.strftime('%Y-%m-%d')
        
        # Save to CSV in the output directory
        output_file = os.path.join(output_dir, f"spotify_news_{year}_{month_name}.csv")
        df.to_csv(output_file, index=False)
        print(f"Data for {month_name} {year} saved to {output_file}")
        
        # Display first few rows
        print(f"\nFirst few rows of the {month_name} {year} data:")
        print(df.head(3))  # Show just 3 rows to keep the output manageable

if __name__ == "__main__":
    main()