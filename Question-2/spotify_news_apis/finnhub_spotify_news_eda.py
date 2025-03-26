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
import glob

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Input directory containing CSV files
data_dir = os.path.join("/Users/armandocriscuolo/c2025/data_science_project_2025/code/Data-Science-Project/Data-Question-2-B/spotify_news_2024_2025_finnhub")

# Output directory for analysis results
output_dir = os.path.join(script_dir, "spotify_news_analysis_results_finnhub")

# Create output directory if it doesn't exist
def create_output_directory():
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir

def load_and_explore_data(data_directory):
    """Load data from all CSV files in the directory and display basic information."""
    if not os.path.exists(data_directory):
        raise FileNotFoundError(f"Data directory not found: {data_directory}")
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(data_directory, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {data_directory}")
    
    print(f"Found {len(csv_files)} CSV files to process.")
    
    # Load and combine all CSV files
    dfs = []
    
    for file in csv_files:
        try:
            month_df = pd.read_csv(file)
            filename = os.path.basename(file)
            print(f"Loaded {filename} with {month_df.shape[0]} rows and {month_df.shape[1]} columns.")
            dfs.append(month_df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Combine all dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Basic data exploration
    print(f"\nCombined dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    print("\nColumn names:", df.columns.tolist())
    print("\nSample data:")
    print(df.head(2))
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
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
    plt.title('Number of Spotify News Articles by Day (2024-2025)')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spotify_news_daily_trend.png'))
    plt.close()
    
    # Monthly aggregation
    if hasattr(df['date'], 'dt'):
        df['month'] = df['date'].dt.strftime('%Y-%m')
        monthly_counts = df.groupby('month').size().sort_index()
        
        plt.figure(figsize=(12, 6))
        monthly_counts.plot(kind='bar', color='skyblue')
        plt.title('Number of Spotify News Articles by Month (2024-2025)')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spotify_news_monthly_trend.png'))
        plt.close()
    
    print(f"Total days with news: {len(daily_counts)}")
    print(f"Average articles per day: {daily_counts.mean():.2f}")
    print(f"Day with most articles: {daily_counts.idxmax()} ({daily_counts.max()} articles)")
    
    # Day of week analysis
    if hasattr(df['date'], 'dt'):
        df['day_of_week'] = df['date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df['day_of_week'].value_counts().reindex(day_order)
        
        plt.figure(figsize=(10, 6))
        day_counts.plot(kind='bar', color='lightgreen')
        plt.title('Distribution of Spotify News Articles by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Articles')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spotify_news_day_of_week.png'))
        plt.close()

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
        plt.close()
    
    # News categories analysis
    if 'category' in df.columns:
        category_counts = df['category'].value_counts()
        print("\nNews categories distribution:")
        print(category_counts)
        
        if len(category_counts) > 1:
            plt.figure(figsize=(10, 6))
            category_counts.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette("pastel"))
            plt.title('Distribution of Spotify News Articles by Category')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'spotify_news_categories.png'))
            plt.close()
            
            # Analyze categories over time
            if hasattr(df['date'], 'dt'):
                df['month'] = df['date'].dt.strftime('%Y-%m')
                category_by_month = pd.crosstab(df['month'], df['category']).sort_index()
                
                plt.figure(figsize=(14, 8))
                category_by_month.plot(kind='bar', stacked=True)
                plt.title('Categories Distribution Over Time')
                plt.xlabel('Month')
                plt.ylabel('Number of Articles')
                plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'spotify_news_categories_over_time.png'))
                plt.close()

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
    stop_words.update(['spotify', 'says', 'said', 'new', 'also', 'may', 'one', 'year', 'first', 'two', 'will']) # Add custom stopwords
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
    plt.title('Top 20 Words in Spotify News Headlines (2024-2025)')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spotify_news_headline_words.png'))
    plt.close()
    
    # Average headline length analysis
    df['headline_length'] = df['headline'].fillna('').apply(len)
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['headline_length'], bins=50, kde=True)
    plt.title('Distribution of Headline Lengths')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.axvline(df['headline_length'].mean(), color='red', linestyle='--', label=f'Mean: {df["headline_length"].mean():.1f}')
    plt.axvline(df['headline_length'].median(), color='green', linestyle='--', label=f'Median: {df["headline_length"].median():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spotify_news_headline_length.png'))
    plt.close()

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
    sns.histplot(df['summary_length'], bins=50, kde=True)
    plt.title('Distribution of Summary Text Length')
    plt.xlabel('Number of Characters')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spotify_news_summary_length.png'))
    plt.close()
    
    # Analyze the beginning of summaries to detect placeholders
    summary_starts = df['summary'].fillna('').apply(lambda x: x[:50] if len(x) > 50 else x)
    common_starts = Counter(summary_starts).most_common(10)
    
    print("\nTop 10 most common summary starts (possible placeholders):")
    for start, count in common_starts:
        # Only print if it appears multiple times
        if count > 1:
            print(f"'{start}...' appears {count} times")
    
    # Check if there are duplicate summaries
    duplicate_summaries = df['summary'].value_counts()
    duplicate_summaries = duplicate_summaries[duplicate_summaries > 1]
    
    if not duplicate_summaries.empty:
        print("\nDuplicate summaries found:")
        for summary, count in duplicate_summaries.items():
            if len(summary) > 50:
                print(f"'{summary[:50]}...' appears {count} times")
            else:
                print(f"'{summary}' appears {count} times")
        
        # Plot duplicates
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(duplicate_summaries)), duplicate_summaries.values)
        plt.title(f'Duplicate Summaries (Found {len(duplicate_summaries)} Different Duplicates)')
        plt.xlabel('Duplicate Summary Index')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spotify_news_duplicate_summaries.png'))
        plt.close()

def analyze_data_completeness(df):
    """Analyze data completeness and quality issues."""
    print("\n=== Data Completeness and Quality Analysis ===")
    
    # Calculate percent of missing values for each column
    missing_values = df.isnull().mean() * 100
    
    print("\nMissing values percentage per column:")
    for column, percentage in missing_values.items():
        print(f"{column}: {percentage:.2f}%")
    
    # Plot missing values
    plt.figure(figsize=(12, 6))
    missing_values.sort_values(ascending=False).plot(kind='bar')
    plt.title('Percentage of Missing Values by Column')
    plt.xlabel('Column')
    plt.ylabel('Missing Values (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spotify_news_missing_values.png'))
    plt.close()
    
    # Check for any potential duplicated rows
    duplicate_rows = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicate_rows}")
    
    # Analyze data consistency across dates
    if 'datetime' in df.columns and 'date' in df.columns:
        # Convert columns to datetime for comparison
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        date_only = df['datetime'].dt.date
        date_from_date_column = pd.to_datetime(df['date']).dt.date
        
        # Check if dates are consistent
        matching_dates = (date_only == date_from_date_column).mean() * 100
        print(f"\nPercentage of rows where 'datetime' and 'date' columns match: {matching_dates:.2f}%")

def analyze_related_tickers(df):
    """Analyze related ticker symbols in the news."""
    print("\n=== Related Ticker Analysis ===")
    
    if 'related' not in df.columns:
        print("No related column found")
        return
    
    # Count occurrences of each ticker
    ticker_counts = df['related'].value_counts()
    
    print("\nTop related tickers:")
    print(ticker_counts.head(15))
    
    # Plot top tickers
    plt.figure(figsize=(12, 6))
    ticker_counts.head(15).plot(kind='bar', color='skyblue')
    plt.title('Top 15 Related Tickers in Spotify News')
    plt.xlabel('Ticker')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spotify_news_related_tickers.png'))
    plt.close()
    
    # Analyze co-occurrence of tickers
    if ticker_counts.shape[0] > 1:  # Only if there's more than one ticker
        print("\nAnalyzing ticker co-occurrences...")
        
        # Extract news with more than one ticker
        df['related_list'] = df['related'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) and ',' in x else [x])
        df_with_multiple = df[df['related_list'].apply(len) > 1]
        
        if not df_with_multiple.empty:
            co_occurrence_count = Counter()
            
            for related_list in df_with_multiple['related_list']:
                # Count all possible pairs in the list
                for i, ticker1 in enumerate(related_list):
                    for ticker2 in related_list[i+1:]:
                        if ticker1 and ticker2:  # Ensure neither ticker is empty
                            pair = tuple(sorted([ticker1.strip(), ticker2.strip()]))
                            co_occurrence_count[pair] += 1
            
            print("\nTop ticker co-occurrences:")
            for (ticker1, ticker2), count in co_occurrence_count.most_common(10):
                print(f"{ticker1} and {ticker2}: {count} articles")

def main():
    """Main function to orchestrate the analysis."""
    print("Starting Exploratory Data Analysis on Spotify News Data\n")
    print("=" * 80)
    
    # Create output directory
    create_output_directory()
    
    # Load and explore the data
    df = load_and_explore_data(data_dir)
    
    # Perform various analyses
    analyze_temporal_patterns(df)
    analyze_sources_and_categories(df)
    analyze_headlines(df)
    analyze_summary_text(df)
    analyze_data_completeness(df)
    analyze_related_tickers(df)
    
    print(f"\nEDA completed. Visualizations have been saved in the '{output_dir}' folder.")

if __name__ == "__main__":
    main() 