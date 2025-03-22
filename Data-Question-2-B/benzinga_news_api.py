import requests
import pandas as pd
import os
import datetime

api_key = 'e73ce990c38845eba7559411d2c78d02'

# If no API key is found, prompt the user
if not api_key:
    api_key = input("Please enter your Benzinga API key: ")

# Benzinga API endpoint for news
url = "https://api.benzinga.com/api/v2/news"

# Define parameters
symbol = "SPOT"  # Spotify stock symbol

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create output directory for the CSV files
output_dir = os.path.join(script_dir, "spotify_news_2024_2025_benzinga")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to fetch news data
def fetch_news_data(symbol, start_date, end_date, api_key):
    params = {
        'token': api_key,
        'pageSize': 100,  # Maximum allowed by Benzinga API
        'displayOutput': 'abstract',  # Use abstract instead of full content
        'dateFrom': start_date,
        'dateTo': end_date,
        'tickers': symbol
    }
    
    headers = {"accept": "application/json"}
    
    all_news = []
    page = 0
    
    # Benzinga API uses pagination, so we need to loop to get all results
    while True:
        params['page'] = page
        try:
            response = requests.get("GET", url, headers=headers, params=params, timeout=30)
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            
            news_data = response.json()
            
            # If no more results, break the loop
            if not news_data:
                break
                
            all_news.extend(news_data)
            page += 1
            print(f"Retrieved page {page} with {len(news_data)} articles")
            
            # Safety check to prevent infinite loops
            if page > 100:
                print("Reached maximum page limit.")
                break
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response content: {e.response.text[:500]}...")  # Print first 500 chars
            return all_news  # Return whatever we've collected so far
    
    return all_news

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
    
    for month in range(3, 13):
        start_date = datetime.date(2024, month, 1)
        if month < 12:
            end_date = datetime.date(2024, month + 1, 1) - datetime.timedelta(days=1)
        else:
            end_date = datetime.date(2024, 12, 31)
        
        date_ranges.append((start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), 2024))
    
    return date_ranges

# Function to process Benzinga news data
def process_benzinga_news(news_data):
    if not news_data:
        return pd.DataFrame()
    
    processed_data = []
    
    for news_item in news_data:
        try:
            # Convert created timestamp to datetime object
            created_datetime = datetime.datetime.strptime(
                news_item.get('created', ''), '%a, %d %b %Y %H:%M:%S %z'
            )
            
            # Extract image URLs
            image_urls = {
                'large': '',
                'small': '',
                'thumb': ''
            }
            
            if 'image' in news_item and news_item['image']:
                for img in news_item['image']:
                    if 'size' in img and 'url' in img:
                        image_urls[img['size']] = img['url']
            
            # Extract channels/categories
            channels = []
            if 'channels' in news_item and news_item['channels']:
                channels = [channel.get('name', '') for channel in news_item['channels']]
            
            # Extract stock tickers
            stocks = []
            if 'stocks' in news_item and news_item['stocks']:
                stocks = [stock.get('name', '') for stock in news_item['stocks']]
            
            # Extract tags
            tags = []
            if 'tags' in news_item and news_item['tags']:
                tags = [tag.get('name', '') for tag in news_item['tags']]
            
            # Get updated timestamp if available
            updated_datetime = None
            if 'updated' in news_item and news_item['updated']:
                try:
                    updated_datetime = datetime.datetime.strptime(
                        news_item.get('updated', ''), '%a, %d %b %Y %H:%M:%S %z'
                    )
                except (ValueError, TypeError):
                    updated_datetime = None
            
            item = {
                'id': news_item.get('id'),
                'datetime': int(created_datetime.timestamp()),
                'date': created_datetime.strftime('%Y-%m-%d'),
                'time': created_datetime.strftime('%H:%M:%S'),
                'headline': news_item.get('title', ''),
                'author': news_item.get('author', ''),
                'source': 'Benzinga',
                'summary': news_item.get('teaser', ''),
                'content_snippet': news_item.get('body', ''),  # With abstract, this will be a snippet rather than full content
                'url': news_item.get('url', ''),
                'image_large': image_urls.get('large', ''),
                'image_small': image_urls.get('small', ''),
                'image_thumb': image_urls.get('thumb', ''),
                'categories': ','.join(channels),
                'related_stocks': ','.join(stocks),
                'tags': ','.join(tags),
                'updated_date': updated_datetime.strftime('%Y-%m-%d %H:%M:%S') if updated_datetime else ''
            }
            processed_data.append(item)
        except Exception as e:
            print(f"Error processing news item: {e}")
            print(f"Problematic item: {news_item.get('id', 'Unknown ID')}")
            continue
    
    return pd.DataFrame(processed_data)

# Main function
def main():
    date_ranges = get_monthly_date_ranges()
    
    for month_info in date_ranges:
        start_date, end_date, year = month_info
        # Extract month number from the start_date
        month_num = int(start_date.split('-')[1])
        month_name = datetime.date(year, month_num, 1).strftime('%B')
        
        print(f"\nFetching Spotify news data from Benzinga for {month_name} {year} ({start_date} to {end_date})...")
        
        news_data = fetch_news_data(symbol, start_date, end_date, api_key)
        
        if not news_data:
            print(f"No data retrieved or an error occurred for {month_name} {year}.")
            continue
        
        print(f"Retrieved {len(news_data)} news articles for {month_name} {year}.")
        
        # Process and convert to DataFrame
        df = process_benzinga_news(news_data)
        
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

if __name__ == "__main__":
    main()
