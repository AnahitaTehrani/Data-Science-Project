#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Science Project: Spotify User Growth Analysis in Europe
This script analyzes Spotify's monthly active user growth in Germany and across Europe,
comparing regional patterns and identifying key growth factors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import os

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create directories for data storage within the script directory
os.makedirs(os.path.join(script_dir, 'output'), exist_ok=True)

#######################
# DATA COLLECTION
#######################

def scrape_spotify_investor_data():
    """
    Scrape data from Spotify Investor Relations.
    In a real implementation, this would parse HTML from the investor relations page.
    """
    print("Collecting data from Spotify Investor Relations...")
    
    # For demonstration purposes, we'll create synthetic data based on known information
    # In a real implementation, this would use requests to get actual data
    
    # Quarterly MAU data (in millions) - approximated from public sources
    quarters = pd.date_range(start='2018-01-01', end='2023-12-31', freq='Q')
    
    # Total global MAU approximated from public sources
    global_mau = [173, 180, 191, 207, 217, 232, 248, 271, 286, 299, 320, 345, 356, 365, 381, 406, 422, 433, 456, 489, 515, 533, 551, 574]
    
    # Europe MAU percentage (approximated) - Europe historically accounts for ~33-35% of Spotify's user base
    europe_percentage = [0.35, 0.34, 0.34, 0.34, 0.34, 0.33, 0.33, 0.33, 0.33, 0.32, 0.32, 0.32, 0.32, 0.31, 0.31, 0.31, 0.31, 0.30, 0.30, 0.30, 0.30, 0.29, 0.29, 0.29]
    
    # Germany as percentage of Europe (approximated) - Germany is one of the largest European markets
    germany_percentage = [0.18, 0.18, 0.19, 0.19, 0.19, 0.20, 0.20, 0.20, 0.21, 0.21, 0.21, 0.22, 0.22, 0.22, 0.22, 0.23, 0.23, 0.23, 0.23, 0.24, 0.24, 0.24, 0.24, 0.25]
    
    # Calculate European and German MAU
    europe_mau = [g * p for g, p in zip(global_mau, europe_percentage)]
    germany_mau = [e * p for e, p in zip(europe_mau, germany_percentage)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': quarters,
        'Global_MAU': global_mau,
        'Europe_MAU': europe_mau,
        'Germany_MAU': germany_mau
    })
    
    return df

def collect_european_market_data():
    """
    Collect data on European markets and their contribution to Spotify growth.
    In a real implementation, this would parse data from multiple sources.
    """
    print("Collecting European market data...")
    
    # For demonstration, creating synthetic data for key European markets
    # Based on information that Europe leads as Spotify's largest market
    countries = ['Germany', 'UK', 'France', 'Italy', 'Spain', 'Sweden', 'Netherlands', 'Poland', 'Other Europe']
    
    # Approximate market share within Europe as of 2023
    market_share = [0.25, 0.22, 0.15, 0.09, 0.08, 0.05, 0.04, 0.03, 0.09]
    
    # Approximate year-over-year growth rate for 2023
    yoy_growth = [0.14, 0.11, 0.13, 0.16, 0.12, 0.08, 0.10, 0.18, 0.12]
    
    # Approximate paid subscriber percentage
    paid_percentage = [0.52, 0.48, 0.45, 0.42, 0.40, 0.65, 0.50, 0.38, 0.44]
    
    # Approximate user engagement score (hours per month)
    engagement_score = [26, 24, 22, 21, 20, 32, 25, 18, 22]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Country': countries,
        'Market_Share': market_share,
        'YoY_Growth': yoy_growth,
        'Paid_Percentage': paid_percentage,
        'Engagement_Score': engagement_score
    })
    
    return df

def collect_growth_factors():
    """
    Collect information on factors influencing Spotify's growth in Europe.
    """
    print("Collecting growth factor data...")
    
    # Key growth factors based on available information
    factors = [
        {
            'Factor': 'Content Expansion',
            'Description': 'Expansion of podcast library and exclusive content',
            'Impact_Score': 8,
            'Germany_Specific': 'Launched German-language original podcasts starting 2019'
        },
        {
            'Factor': 'Premium Features',
            'Description': 'Addition of high-quality audio options and UI improvements',
            'Impact_Score': 7,
            'Germany_Specific': 'Higher than average premium conversion rate in Germany'
        },
        {
            'Factor': 'Market Penetration',
            'Description': 'Strategic marketing and partnerships with telecom providers',
            'Impact_Score': 9,
            'Germany_Specific': 'Partnership with Deutsche Telekom boosted adoption'
        },
        {
            'Factor': 'Competitive Landscape',
            'Description': 'Response to competition from Apple Music, Amazon, and local services',
            'Impact_Score': 6,
            'Germany_Specific': 'Competed with local services like Deezer and Amazon Music'
        },
        {
            'Factor': 'Regional Pricing Strategy',
            'Description': 'Adjusted subscription prices based on market conditions',
            'Impact_Score': 7,
            'Germany_Specific': 'Premium family plan particularly successful in German market'
        }
    ]
    
    return pd.DataFrame(factors)

#######################
# DATA ANALYSIS
#######################

def analyze_user_growth(df):
    """
    Analyze the growth trends in Spotify's user base.
    """
    print("Analyzing user growth trends...")
    
    # Calculate growth rates
    df['Global_QoQ_Growth'] = df['Global_MAU'].pct_change() * 100
    df['Europe_QoQ_Growth'] = df['Europe_MAU'].pct_change() * 100
    df['Germany_QoQ_Growth'] = df['Germany_MAU'].pct_change() * 100
    
    # Calculate CAGR (Compound Annual Growth Rate)
    years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
    global_cagr = (df['Global_MAU'].iloc[-1] / df['Global_MAU'].iloc[0]) ** (1/years) - 1
    europe_cagr = (df['Europe_MAU'].iloc[-1] / df['Europe_MAU'].iloc[0]) ** (1/years) - 1
    germany_cagr = (df['Germany_MAU'].iloc[-1] / df['Germany_MAU'].iloc[0]) ** (1/years) - 1
    
    # Calculate Europe's share of global MAU over time
    df['Europe_Share'] = df['Europe_MAU'] / df['Global_MAU'] * 100
    
    # Calculate Germany's share of European MAU over time
    df['Germany_Share_of_Europe'] = df['Germany_MAU'] / df['Europe_MAU'] * 100
    
    growth_metrics = {
        'Global_CAGR': global_cagr * 100,
        'Europe_CAGR': europe_cagr * 100,
        'Germany_CAGR': germany_cagr * 100,
        'Europe_Share_Start': df['Europe_Share'].iloc[0],
        'Europe_Share_End': df['Europe_Share'].iloc[-1],
        'Germany_Share_Start': df['Germany_Share_of_Europe'].iloc[0],
        'Germany_Share_End': df['Germany_Share_of_Europe'].iloc[-1]
    }
    
    return df, growth_metrics

def analyze_european_markets(market_df):
    """
    Analyze the contribution and patterns of different European markets.
    """
    print("Analyzing European market contributions...")
    
    # Calculate weighted growth contribution
    market_df['Growth_Contribution'] = market_df['Market_Share'] * market_df['YoY_Growth']
    
    # Rank countries by different metrics
    market_df['Market_Share_Rank'] = market_df['Market_Share'].rank(ascending=False)
    market_df['Growth_Rank'] = market_df['YoY_Growth'].rank(ascending=False)
    market_df['Engagement_Rank'] = market_df['Engagement_Score'].rank(ascending=False)
    
    # Create composite score for overall importance
    market_df['Importance_Score'] = (
        market_df['Market_Share'] * 0.5 +
        market_df['YoY_Growth'] * 0.3 +
        market_df['Paid_Percentage'] * 0.2
    )
    
    return market_df.sort_values('Importance_Score', ascending=False)

def analyze_germany_comparison(market_df):
    """
    Analyze how Germany compares to other European markets.
    """
    print("Comparing Germany to other European markets...")
    
    # Extract Germany data
    germany_data = market_df[market_df['Country'] == 'Germany'].iloc[0]
    
    # Calculate how Germany compares to European average
    european_avg = market_df[market_df['Country'] != 'Other Europe'].mean(numeric_only=True)
    
    comparison = {
        'Market_Share_vs_Avg': (germany_data['Market_Share'] / european_avg['Market_Share'] - 1) * 100,
        'Growth_vs_Avg': (germany_data['YoY_Growth'] / european_avg['YoY_Growth'] - 1) * 100,
        'Paid_Percentage_vs_Avg': (germany_data['Paid_Percentage'] / european_avg['Paid_Percentage'] - 1) * 100,
        'Engagement_vs_Avg': (germany_data['Engagement_Score'] / european_avg['Engagement_Score'] - 1) * 100
    }
    
    return comparison

#######################
# VISUALIZATION
#######################

def plot_user_growth(growth_df):
    """
    Create visualizations for user growth trends.
    """
    print("Creating user growth visualizations...")
    
    # Get script directory for proper file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Plot MAU trends
    plt.figure(figsize=(12, 6))
    plt.plot(growth_df['Date'], growth_df['Global_MAU'], label='Global MAU', linewidth=2)
    plt.plot(growth_df['Date'], growth_df['Europe_MAU'], label='Europe MAU', linewidth=2)
    plt.plot(growth_df['Date'], growth_df['Germany_MAU'], label='Germany MAU', linewidth=2)
    plt.title('Spotify Monthly Active Users Growth (2018-2023)', fontsize=16)
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Monthly Active Users (millions)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, 'output', 'mau_growth_trends.png'), dpi=300, bbox_inches='tight')
    
    # Plot Europe and Germany share trends
    plt.figure(figsize=(12, 6))
    plt.plot(growth_df['Date'], growth_df['Europe_Share'], label='Europe % of Global', linewidth=2)
    plt.plot(growth_df['Date'], growth_df['Germany_Share_of_Europe'], label='Germany % of Europe', linewidth=2)
    plt.title('Shifting Regional Contributions to Spotify User Base (2018-2023)', fontsize=16)
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Percentage Share (%)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, 'output', 'regional_share_trends.png'), dpi=300, bbox_inches='tight')

def plot_european_markets(market_df):
    """
    Create visualizations for European market analysis.
    """
    print("Creating European market visualizations...")
    
    # Get script directory for proper file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create subset without "Other Europe" for cleaner visualization
    plot_df = market_df[market_df['Country'] != 'Other Europe'].sort_values('Market_Share', ascending=False)
    
    # Plot market share by country
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Country', y='Market_Share', data=plot_df)
    plt.title('Spotify Market Share by European Country (2023)', fontsize=16)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Share of European Market', fontsize=12)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(script_dir, 'output', 'europe_market_share.png'), dpi=300, bbox_inches='tight')
    
    # Create scatter plot of growth vs. engagement
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='YoY_Growth', 
        y='Engagement_Score', 
        size='Market_Share',
        sizes=(100, 1000),
        alpha=0.7,
        data=plot_df
    )
    
    # Add country labels to the points
    for i, row in plot_df.iterrows():
        plt.annotate(
            row['Country'], 
            (row['YoY_Growth'], row['Engagement_Score']),
            fontsize=9,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.title('European Markets: Growth vs. Engagement (2023)', fontsize=16)
    plt.xlabel('Year-over-Year Growth Rate', fontsize=12)
    plt.ylabel('User Engagement (Hours per Month)', fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, 'output', 'europe_growth_engagement.png'), dpi=300, bbox_inches='tight')

def plot_growth_factors(factors_df):
    """
    Visualize the impact of different growth factors.
    """
    print("Creating growth factors visualization...")
    
    # Get script directory for proper file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Impact_Score', y='Factor', data=factors_df, palette='viridis')
    plt.title('Factors Influencing Spotify Growth in Europe', fontsize=16)
    plt.xlabel('Impact Score (1-10)', fontsize=12)
    plt.ylabel('Growth Factor', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'output', 'growth_factors.png'), dpi=300, bbox_inches='tight')

#######################
# MAIN EXECUTION
#######################

def main():
    """
    Main execution function to run the analysis.
    """
    print("Starting Spotify European Growth Analysis...")
    
    # Get script directory for proper file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Data Collection
    growth_data = scrape_spotify_investor_data()
    market_data = collect_european_market_data()
    factor_data = collect_growth_factors()
    
    # Data Analysis
    growth_data, growth_metrics = analyze_user_growth(growth_data)
    market_analysis = analyze_european_markets(market_data)
    germany_comparison = analyze_germany_comparison(market_data)
    
    # Visualizations
    plot_user_growth(growth_data)
    plot_european_markets(market_analysis)
    plot_growth_factors(factor_data)
    
    # Print key findings
    print("\n===== KEY FINDINGS =====")
    print("\nQuestion 1: Spotify's Monthly Active User Growth in Germany and Europe")
    print(f"- Global CAGR (2018-2023): {growth_metrics['Global_CAGR']:.2f}%")
    print(f"- Europe CAGR (2018-2023): {growth_metrics['Europe_CAGR']:.2f}%")
    print(f"- Germany CAGR (2018-2023): {growth_metrics['Germany_CAGR']:.2f}%")
    print(f"- Europe's share of global MAU decreased from {growth_metrics['Europe_Share_Start']:.2f}% to {growth_metrics['Europe_Share_End']:.2f}%")
    print(f"- Germany's share of European MAU increased from {growth_metrics['Germany_Share_Start']:.2f}% to {growth_metrics['Germany_Share_End']:.2f}%")
    print("- Key growth factors include content expansion, premium features, and strategic partnerships")
    
    print("\nQuestion 2: European Markets Driving Spotify's Growth")
    print("- Top European markets by importance: ")
    for i, row in market_analysis.head(3).iterrows():
        print(f"  {row['Country']}: {row['Market_Share']*100:.1f}% market share, {row['YoY_Growth']*100:.1f}% growth")
    
    print("\nGermany's Performance Compared to European Average:")
    print(f"- Market Share: {germany_comparison['Market_Share_vs_Avg']:.1f}% higher than European average")
    print(f"- Growth Rate: {germany_comparison['Growth_vs_Avg']:.1f}% {'higher' if germany_comparison['Growth_vs_Avg'] > 0 else 'lower'} than European average")
    print(f"- Paid Subscription Rate: {germany_comparison['Paid_Percentage_vs_Avg']:.1f}% higher than European average")
    print(f"- User Engagement: {germany_comparison['Engagement_vs_Avg']:.1f}% higher than European average")
    
    print("\nVisualization outputs saved to the 'output' directory.")
    
    # Save results to CSV for further analysis
    growth_data.to_csv(os.path.join(script_dir, 'output', 'spotify_growth_data.csv'), index=False)
    market_analysis.to_csv(os.path.join(script_dir, 'output', 'european_market_analysis.csv'), index=False)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
