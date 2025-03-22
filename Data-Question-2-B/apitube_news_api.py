import requests
import pandas as pd
import os
import datetime
import argparse
import json

api_key = 'api_live_ZDxwA5X5C55cdICpOW0Zcy8Tem3zhAd5Epr0YyurvDhHdKm0'

# If no API key is found, prompt the user
if not api_key:
    api_key = input("Please enter your APITube API key: ")

# APITube API endpoint for news search
url = "https://api.apitube.io/v1/news/everything"

# Define parameters
symbol = "SPOT"  # Spotify stock symbol

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create output directory for the CSV files
output_dir = os.path.join(script_dir, "spotify_news_2024_2025_apitube")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to fetch news data
def fetch_news_data(symbol, start_date, end_date, api_key, test_mode=False):
    # Simplified query for testing - just search for Spotify
    # For non-test mode, use a more targeted query for relevant business articles
    if test_mode:
        # Just use the basic keyword for wider coverage during testing
        lucene_query = "Spotify"
    else:
        lucene_query = f'Spotify AND (business OR earnings OR revenue OR stock OR financial OR "quarterly results" OR CEO OR investors)'
    
    # Print the query and date range being used
    print(f"Using query: {lucene_query}")
    print(f"Date range: {start_date} to {end_date}")
    
    params = {
        'q': lucene_query,
        'published_at.start': start_date,
        'published_at.end': end_date,
        'language': 'en',
        'sort_by': 'published_at',
        'sort_order': 'desc',
        'per_page': 5 if test_mode else 100,  # Limit results when testing
        'page': 1,
        'api_key': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        data = response.json()
        
        # Print the full response for inspection
        print("\nFULL API RESPONSE:")
        print(json.dumps(data, indent=2))
        
        # For debugging - save the raw response to a JSON file
        if test_mode:
            with open(os.path.join(output_dir, "test_response.json"), "w") as f:
                json.dump(data, f, indent=2)
            print(f"Raw response saved to {os.path.join(output_dir, 'test_response.json')}")
        
        # Check if we have valid data in the response
        # Updated to match the correct response structure
        if 'results' in data:
            total_count = data.get('total_count', 0)
            print(f"Total articles found: {total_count}")
            return data['results']
        else:
            print(f"No articles found in the response: {data}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response content: {e.response.text[:500]}...")  # Print first 500 chars
        return []

# Function to get date ranges for months in 2024 and 2025
def get_monthly_date_ranges():
    date_ranges = []
    
    # January, February, and March 2025 (since we're currently in March 2025)
    for month in range(1, 4):
        start_date = datetime.date(2025, month, 1)
        if month < 12:
            end_date = datetime.date(2025, month + 1, 1) - datetime.timedelta(days=1)
        else:
            end_date = datetime.date(2025, 12, 31)
        
        date_ranges.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 2025))
    
    # March to December 2024
    for month in range(3, 13):
        start_date = datetime.date(2024, month, 1)
        if month < 12:
            end_date = datetime.date(2024, month + 1, 1) - datetime.timedelta(days=1)
        else:
            end_date = datetime.date(2024, 12, 31)
        
        date_ranges.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 2024))
    
    return date_ranges

# For test mode, use a historical date range from 2023 to see if API works at all
def get_test_date_range():
    # Use January and February 2023 for testing
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2023, 2, 28)
    
    return [(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 2023)]

# Function to process APITube news data
def process_apitube_news(news_data):
    if not news_data:
        return pd.DataFrame()
    
    processed_data = []
    
    for news_item in news_data:
        try:
            # Extract published date - updated field name based on actual response
            published_date = news_item.get('published_at', '')
            if published_date:
                published_datetime = datetime.datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                unix_timestamp = int(published_datetime.timestamp())
                formatted_date = published_datetime.strftime('%Y-%m-%d')
                formatted_time = published_datetime.strftime('%H:%M:%S')
            else:
                unix_timestamp = 0
                formatted_date = ''
                formatted_time = ''
            
            # Extract source information - adapt to actual response structure
            source_name = news_item.get('source_name', 'Unknown')
            
            # Extract categories - adapt to actual response structure
            categories = []
            if 'categories' in news_item and news_item['categories']:
                for category in news_item['categories']:
                    categories.append(category)
            
            # Create item dictionary - adapted to match actual response structure
            item = {
                'datetime': unix_timestamp,
                'date': formatted_date,
                'time': formatted_time,
                'headline': news_item.get('title', ''),
                'author': news_item.get('author', ''),
                'source': source_name,
                'summary': news_item.get('description', ''),
                'content_snippet': news_item.get('body', ''),
                'url': news_item.get('url', ''),
                'image_url': news_item.get('image_url', ''),
                'categories': ','.join(categories),
                'related_stocks': symbol,  # We're searching for this stock specifically
                'tags': news_item.get('keywords', ''),
                'sentiment': news_item.get('sentiment', 'neutral')
            }
            
            processed_data.append(item)
        except Exception as e:
            print(f"Error processing news item: {e}")
            print(f"Problematic item: {news_item.get('title', 'Unknown')}")
            continue
    
    return pd.DataFrame(processed_data)

# Main function
def main():
    parser = argparse.ArgumentParser(description='Fetch Spotify news from APITube')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited requests')
    args = parser.parse_args()
    
    # Use test date range if in test mode, otherwise use full monthly ranges
    if args.test:
        print("Running in TEST MODE - limited results will be fetched")
        date_ranges = get_test_date_range()
    else:
        date_ranges = get_monthly_date_ranges()
    
    for month_info in date_ranges:
        start_date, end_date, year = month_info
        # Extract month number from the start_date
        month_num = int(start_date.split('-')[1])
        month_name = datetime.date(year, month_num, 1).strftime('%B')
        
        print(f"\nFetching Spotify business news from APITube for {month_name} {year} ({start_date} to {end_date})...")
        
        news_data = fetch_news_data(symbol, start_date, end_date, api_key, args.test)
        
        if not news_data:
            print(f"No data retrieved or an error occurred for {month_name} {year}.")
            continue
        
        print(f"Retrieved {len(news_data)} news articles for {month_name} {year}.")
        
        # Process and convert to DataFrame
        df = process_apitube_news(news_data)
        
        # Check if we received any data
        if df.empty:
            print(f"No news data available for {month_name} {year}.")
            continue
        
        # Save to CSV in the output directory
        output_file = os.path.join(output_dir, f"spotify_news_{year}_{month_name}.csv")
        df.to_csv(output_file, index=False)
        print(f"Data for {month_name} {year} saved to {output_file}")
        
        # Display first few rows
        print(f"\nFirst few rows of the {month_name} {year} data:")
        print(df.head(3))  # Show just 3 rows to keep the output manageable
        
        # Break after first iteration if in test mode
        if args.test:
            print("Test completed. Run without --test flag to fetch full data.")
            break

if __name__ == "__main__":
    main() 