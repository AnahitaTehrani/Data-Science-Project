import requests
import pandas as pd
from bs4 import BeautifulSoup
from textblob import TextBlob
import time
import re
import random
from tqdm import tqdm
import urllib3
from newsapi import NewsApiClient
from datetime import datetime, timedelta

# Initialize client
newsapi = NewsApiClient(api_key='b28ab6db6fbe4dc98543806f04d24e3f')

# Define a 30-day period in 2025
end_date = '2025-03-19'  # 
start_date = '2025-02-20'  # 30 days before the end date

# More specific Spotify-related query
query = 'Spotify AND (music OR streaming OR podcast OR premium OR playlists OR artists OR "new feature" OR update)'

# Get articles
data = newsapi.get_everything(
    q=query,
    from_param=start_date,
    to=end_date,
    language='en',
    sort_by='relevancy',
    page_size=100  # Limit to 10 articles
)

# Funktion zur Textextraktion
def extract_text_from_url(url):
    try:
        # Use a rotating set of user agents to avoid being blocked
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
        ]
        
        headers = {'User-Agent': random.choice(user_agents)}
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            # Try to determine encoding
            if 'charset' in response.headers.get('content-type', '').lower():
                encoding = response.encoding
            else:
                encoding = response.apparent_encoding
                
            soup = BeautifulSoup(response.content.decode(encoding, errors='replace'), 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'meta', 'head', 'link']):
                element.extract()
            
            # Look for main content containers first
            main_content = soup.find(['main', 'article', 'div', 'section'], class_=lambda x: x and ('content' in x.lower() or 'article' in x.lower() or 'main' in x.lower()))
            
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                # Fallback to the whole body if no main content found
                text = soup.get_text(separator=' ', strip=True)
            
            # Clean text - remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Try to remove common non-content text
            patterns_to_remove = [
                r'cookie policy',
                r'privacy policy',
                r'terms of service',
                r'accept cookies',
                r'advertisement',
                r'subscribe',
                r'newsletter',
                r'sign up',
                r'copyright',
                r'all rights reserved',
            ]
            
            for pattern in patterns_to_remove:
                text = re.sub(r'(?i)' + pattern + r'[^\n.]*[\n.]', ' ', text)
            
            return text
        else:
            return f"Failed to retrieve content: Status code {response.status_code}"
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Funktion zur Sentiment-Analyse
def analyze_sentiment(text):
    if isinstance(text, str) and len(text) > 100:  # Only analyze if we have enough text
        try:
            analysis = TextBlob(text)
            # Get sentiment polarity (-1 to 1, where -1 is negative, 0 is neutral, 1 is positive)
            polarity = analysis.sentiment.polarity
            # Get sentiment subjectivity (0 to 1, where 0 is objective, 1 is subjective)
            subjectivity = analysis.sentiment.subjectivity
            
            # Determine sentiment category
            if polarity > 0.1:
                sentiment = "Positive"
            elif polarity < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
                
            return {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "sentiment": sentiment
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {
                "polarity": 0,
                "subjectivity": 0,
                "sentiment": "Error"
            }
    else:
        return {
            "polarity": 0,
            "subjectivity": 0,
            "sentiment": "Insufficient Text"
        }

# Liste für die Ergebnisse
results = []

# Verarbeite jeden Artikel
print(f"Found {len(data.get('articles', []))} articles to process")
print("Processing articles and performing sentiment analysis...\n")

# Use tqdm for progress bar
for article in tqdm(data.get("articles", []), desc="Processing Articles"):
    title = article['title']
    published_date = article['publishedAt']
    article_url = article['url']
    
    # Skip problematic URLs
    skip_domains = ['github.com', 'youtube.com', 'facebook.com', 'twitter.com']
    if any(domain in article_url for domain in skip_domains):
        print(f"Skipping {article_url} (domain in skip list)")
        continue
    
    # Extrahiere Text
    content = extract_text_from_url(article_url)
    
    # Skip if extraction failed
    if content.startswith("Error") or content.startswith("Failed"):
        print(f"Skipping {article_url} ({content})")
        continue
    
    # Kürze Text für die Ausgabe
    content_preview = content[:300] + "..." if len(content) > 300 else content
    
    # Analysiere Sentiment
    sentiment_results = analyze_sentiment(content)
    
    # Speichere Ergebnisse
    results.append({
        "title": title,
        "published_date": published_date,
        "url": article_url,
        "content_preview": content_preview,
        "polarity": sentiment_results["polarity"],
        "subjectivity": sentiment_results["subjectivity"],
        "sentiment": sentiment_results["sentiment"]
    })
    
    # Random delay between requests to avoid getting blocked
    time.sleep(random.uniform(1.5, 3.5))

# Erstelle DataFrame
df = pd.DataFrame(results)

# Zeige Ergebnisse an
print("\n=== Sentiment Analysis Results ===")
for i, row in df.iterrows():
    print(f"\n{i+1}. Title: {row['title']}")
    print(f"   Published: {row['published_date']}")
    print(f"   Sentiment: {row['sentiment']} (Polarity: {row['polarity']:.2f}, Subjectivity: {row['subjectivity']:.2f})")

# Berechne Sentiment-Statistiken
sentiment_counts = df['sentiment'].value_counts()
print("\n=== Sentiment Distribution ===")
print(sentiment_counts)

# Berechne durchschnittliche Polarität und Subjektivität
print(f"\nAverage Polarity: {df['polarity'].mean():.2f}")
print(f"Average Subjectivity: {df['subjectivity'].mean():.2f}")

# Speichere Ergebnisse in CSV
df.to_csv('spotify_stock_news_sentiment.csv', index=False)
print("\nResults saved to spotify_stock_news_sentiment.csv")