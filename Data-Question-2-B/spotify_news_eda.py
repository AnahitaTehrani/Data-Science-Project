import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# File path
csv_file = os.path.join(script_dir, "spotify_news_2024_3.csv")

# Output directory
output_dir = os.path.join(script_dir, "spotify_analysis_output_3")

# Create output directory if it doesn't exist
def create_output_directory():
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def load_and_explore_data(file_path):
    """Load the data and display basic information."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}. Please ensure the file exists in the same directory as this script.")
    
    df = pd.read_csv(file_path)
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def analyze_temporal_patterns(df):
    """Analyze news distribution over time."""
    print("\n=== Temporal Analysis ===")
    
    # Convert date to datetime if it's not already
    if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Group by date and count articles
    daily_counts = df.groupby(df['date']).size()
    
    # Plot daily article counts
    plt.figure(figsize=(14, 6))
    daily_counts.plot(kind='line', marker='o', alpha=0.7)
    plt.title('Number of Spotify News Articles by Day (2024)')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spotify_news_daily_trend.png'))
    
    # Monthly aggregation
    if hasattr(df['date'], 'dt'):
        df['month'] = df['date'].dt.strftime('%Y-%m')
        monthly_counts = df.groupby('month').size()
        
        plt.figure(figsize=(12, 6))
        monthly_counts.plot(kind='bar', color='skyblue')
        plt.title('Number of Spotify News Articles by Month (2024)')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spotify_news_monthly_trend.png'))
    
    print(f"Total days with news: {len(daily_counts)}")
    print(f"Average articles per day: {daily_counts.mean():.2f}")
    print(f"Day with most articles: {daily_counts.idxmax()} ({daily_counts.max()} articles)")

def analyze_sources_and_categories(df):
    """Analyze news sources and categories."""
    print("\n=== Source and Category Analysis ===")
    
    # News sources analysis
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        print("\nTop 10 news sources:")
        print(source_counts.head(10))
        
        plt.figure(figsize=(12, 6))
        source_counts.head(10).plot(kind='bar', color='lightgreen')
        plt.title('Top 10 Sources of Spotify News Articles')
        plt.xlabel('Source')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spotify_news_top_sources.png'))
    
    # News categories analysis
    if 'category' in df.columns:
        category_counts = df['category'].value_counts()
        print("\nNews categories distribution:")
        print(category_counts)
        if category_counts > 1:    
            plt.figure(figsize=(10, 6))
            category_counts.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette("pastel"))
            plt.title('Distribution of Spotify News Articles by Category')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'spotify_news_categories.png'))

def analyze_headlines(df):
    """Analyze news headlines for common words and sentiment."""
    print("\n=== Headline Analysis ===")
    
    if 'headline' not in df.columns:
        print("No headline column found")
        return
    
    # Download NLTK resources if needed
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
    
    # Get all headlines as a single string
    all_headlines = ' '.join(df['headline'].dropna())
    
    # Tokenize and clean text
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(all_headlines.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_freq = Counter(filtered_tokens)
    
    # Print most common words
    print("\nMost common words in headlines:")
    for word, count in word_freq.most_common(20):
        print(f"{word}: {count}")
    
    # Plot word frequencies
    plt.figure(figsize=(14, 8))
    words = [word for word, count in word_freq.most_common(20)]
    counts = [count for word, count in word_freq.most_common(20)]
    
    sns.barplot(x=counts, y=words)
    plt.title('Top 20 Words in Spotify News Headlines (2024)')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spotify_news_headline_words.png'))

def analyze_summary_text(df):
    """Analyze the summary text of the news articles."""
    print("\n=== Summary Text Analysis ===")
    
    if 'summary' not in df.columns:
        print("No summary column found")
        return
    
    # Check summary length distribution
    df['summary_length'] = df['summary'].fillna('').apply(len)
    
    print(f"\nAverage summary length: {df['summary_length'].mean():.2f} characters")
    print(f"Median summary length: {df['summary_length'].median()} characters")
    print(f"Min summary length: {df['summary_length'].min()} characters")
    print(f"Max summary length: {df['summary_length'].max()} characters")
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['summary_length'], bins=30, kde=True)
    plt.title('Distribution of Summary Text Length')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spotify_news_summary_length.png'))

def main():
    """Main function to orchestrate the analysis."""
    print("Starting Exploratory Data Analysis on Spotify News Data\n")
    print("=" * 80)
    
    # Create output directory
    create_output_directory()
    
    # Load and explore the data
    df = load_and_explore_data(csv_file)
    
    # Perform various analyses
    analyze_temporal_patterns(df)
    analyze_sources_and_categories(df)
    analyze_headlines(df)
    analyze_summary_text(df)
    
    print(f"\nEDA completed. Visualizations have been saved in the '{output_dir}' folder.")

if __name__ == "__main__":
    main() 