import requests
import json
import csv
import os
import datetime

url = "https://api.apitube.io/v1/news/everything"

querystring = {"per_page":"1",
               "api_key":"",
               "title":"Spotify",
               "body":"Spotify",
               "language":"en",
               "published_at.start":"2024-01-01",
               "published_at.end":"2024-02-01",
               "is_duplicate":"false",
               "category.name":"Music",         
               }

response = requests.request("GET", url, params=querystring)

# Parse the JSON response and print it with nice formatting
parsed_response = json.loads(response.text)
print(json.dumps(parsed_response, indent=2))

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create output directory for the CSV files
csv_dir = os.path.join(script_dir, "api_tube_news_data")
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# Create a unique filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(csv_dir, f"api_tube_news_data_{timestamp}.csv")

# Save the data to a CSV file
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    # Define CSV headers based on the structure of the API response
    fieldnames = [
        'id', 
        'published_at', 
        'title', 
        'description', 
        'body',
        'categories',
        'industries',
        'entities',
        'topics',
        'language',
        'sentiment_overall_score',
        'sentiment_overall_polarity',
        'sentiment_title_score',
        'sentiment_title_polarity',
        'sentiment_body_score',
        'sentiment_body_polarity',
        'summary'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    # Extract relevant data from each news item
    for news_item in parsed_response.get('results', []):
        # Process sentiment data
        sentiment = news_item.get('sentiment', {})
        overall = sentiment.get('overall', {})
        title_sentiment = sentiment.get('title', {})
        body_sentiment = sentiment.get('body', {})
        
        # Process summary data - convert list of summary sentences to a single string
        summary_list = news_item.get('summary', [])
        summary_text = ' '.join([item.get('sentence', '') for item in summary_list]) if summary_list else ''
        
        # Process categories, industries, entities, topics - handle as JSON strings to avoid type errors
        categories = json.dumps(news_item.get('categories')) if news_item.get('categories') else ''
        industries = json.dumps(news_item.get('industries')) if news_item.get('industries') else ''
        entities = json.dumps(news_item.get('entities')) if news_item.get('entities') else ''
        topics = json.dumps(news_item.get('topics')) if news_item.get('topics') else ''
        
        writer.writerow({
            'id': news_item.get('id', ''),
            'published_at': news_item.get('published_at', ''),
            'title': news_item.get('title', ''),
            'description': news_item.get('description', ''),
            'body': news_item.get('body', ''),
            'categories': categories,
            'industries': industries,
            'entities': entities,
            'topics': topics,
            'language': news_item.get('language', ''),
            'sentiment_overall_score': overall.get('score', ''),
            'sentiment_overall_polarity': overall.get('polarity', ''),
            'sentiment_title_score': title_sentiment.get('score', ''),
            'sentiment_title_polarity': title_sentiment.get('polarity', ''),
            'sentiment_body_score': body_sentiment.get('score', ''),
            'sentiment_body_polarity': body_sentiment.get('polarity', ''),
            'summary': summary_text
        })

print(f"Data saved to {csv_filename}")