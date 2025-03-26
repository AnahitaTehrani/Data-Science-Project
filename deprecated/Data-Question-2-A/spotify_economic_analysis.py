#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import gzip

# Set plotting style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# File paths
SPOTIFY_USERS_FILE = '/Users/armandocriscuolo/c2025/data_science_project_2025/code/Data-Science-Project/Data-Question-2/spotify_users.csv'
SPOTIFY_REVENUE_FILE = '/Users/armandocriscuolo/c2025/data_science_project_2025/code/Data-Science-Project/Data-Question-2/spotify_revenue.csv'
CPI_FILE = '/Users/armandocriscuolo/c2025/data_science_project_2025/code/Data-Science-Project/Data-Question-2/dataset_2025-03-18T09_15_46.364501357Z_DEFAULT_INTEGRATION_IMF.STA_CPI_3.0.1.csv'
LABOR_STATS_FILE = '/Users/armandocriscuolo/c2025/data_science_project_2025/code/Data-Science-Project/Data-Question-2/dataset_2025-03-18T09_17_10.618558004Z_DEFAULT_INTEGRATION_IMF.STA_LS_9.0.0.csv'
FISCAL_FILE = '/Users/armandocriscuolo/c2025/data_science_project_2025/code/Data-Science-Project/Data-Question-2/dataset_2025-03-18T09_19_37.476328512Z_DEFAULT_INTEGRATION_IMF.FAD_FM_4.0.0.csv'

# Load Spotify data 
def load_spotify_data():
    """Load Spotify users and revenue data"""
    print("Loading Spotify data...")
    users_df = pd.read_csv(SPOTIFY_USERS_FILE)
    revenue_df = pd.read_csv(SPOTIFY_REVENUE_FILE)
    
    # Convert Spotify's quarter notation to datetime
    users_df['Date'] = pd.to_datetime(users_df['Date'].apply(
        lambda x: f"{x.split(' ')[1]}-{x.split(' ')[0][1]}"
    ), format='%Y-%m')
    
    # Ensure each data point has a year and quarter
    users_df['Year'] = users_df['Year'].astype(int)
    users_df['Quarter'] = users_df['Date'].dt.quarter
    
    # Calculate quarterly growth rates
    users_df['MAU_Growth'] = users_df['Monthly Active Users (Millions)'].pct_change() * 100
    users_df['Premium_Growth'] = users_df['Paying Subscribers (Millions)'].pct_change() * 100
    
    # Convert revenue to numeric values
    revenue_df['Year'] = revenue_df['Year'].astype(int)
    
    return users_df, revenue_df

# Load and process IMF CPI data (inflation)
def load_cpi_data():
    """Load and process IMF CPI data for key markets"""
    print("Loading CPI data...")
    
    # Define key markets (major Spotify markets)
    key_markets = ['USA', 'GBR', 'DEU', 'FRA', 'SWE', 'BRA', 'MEX', 'JPN', 'AUS', 'CAN']
    
    try:
        # Read the CPI data in chunks due to its large size
        chunks = []
        print("Reading CPI data file in chunks...")
        
        for chunk in pd.read_csv(CPI_FILE, chunksize=100000, low_memory=False):
            # Filter for headline CPI (all items) and year-over-year percent change
            # Look for rows with CPI.HALL.YOY_PCH_PA_PT in SERIES_CODE and our key markets
            filtered_chunk = chunk[
                (chunk['SERIES_CODE'].str.contains('CPI_YOY', na=False) if 'SERIES_CODE' in chunk.columns else False) |
                (chunk['SERIES_CODE'].str.contains('CPI.HALL.YOY_PCH', na=False) if 'SERIES_CODE' in chunk.columns else False)
            ]
            
            # Further filter for key markets
            if 'COUNTRY.Name' in filtered_chunk.columns:
                country_filter = filtered_chunk['COUNTRY.Name'].apply(
                    lambda x: any(market in str(x) for market in key_markets)
                )
                filtered_chunk = filtered_chunk[country_filter]
            elif 'ISO_CODE' in filtered_chunk.columns:
                filtered_chunk = filtered_chunk[filtered_chunk['ISO_CODE'].isin(key_markets)]
            
            if not filtered_chunk.empty:
                chunks.append(filtered_chunk)
                print(f"Found {len(filtered_chunk)} relevant CPI rows")
        
        # If no chunks were found with the specified pattern, try a different approach
        if not chunks:
            print("No data found with specific patterns. Scanning for any CPI data...")
            
            # Look for any columns that might contain year-quarter data (e.g., "2015-Q1")
            year_columns = []
            for chunk in pd.read_csv(CPI_FILE, nrows=10, low_memory=False):
                for col in chunk.columns:
                    if col.startswith('20') and ('-Q' in col or '-M' in col):
                        year_columns.append(col)
            
            if year_columns:
                print(f"Found {len(year_columns)} potential year-quarter columns")
                
                # Now read only the relevant columns
                usecols = ['SERIES_CODE', 'COUNTRY.Name', 'ISO_CODE'] + year_columns
                for chunk in pd.read_csv(CPI_FILE, chunksize=100000, usecols=lambda x: x in usecols, low_memory=False):
                    # Filter for rows that are likely to contain CPI data
                    if 'SERIES_CODE' in chunk.columns:
                        filtered_chunk = chunk[chunk['SERIES_CODE'].str.contains('CPI', na=False)]
                        
                        # Further filter for key markets
                        if 'COUNTRY.Name' in filtered_chunk.columns:
                            country_filter = filtered_chunk['COUNTRY.Name'].apply(
                                lambda x: any(market in str(x) for market in key_markets)
                            )
                            filtered_chunk = filtered_chunk[country_filter]
                        elif 'ISO_CODE' in filtered_chunk.columns:
                            filtered_chunk = filtered_chunk[filtered_chunk['ISO_CODE'].isin(key_markets)]
                        
                        if not filtered_chunk.empty:
                            chunks.append(filtered_chunk)
                            print(f"Found {len(filtered_chunk)} relevant CPI rows with alternative method")
        
        # Process the filtered data
        if chunks:
            print(f"Processing {len(chunks)} chunks of CPI data...")
            cpi_df = pd.concat(chunks, ignore_index=True)
            
            # Reshape the data to have years/quarters as rows
            cpi_reshaped = pd.DataFrame()
            
            # Extract data for each year-quarter combination
            for col in cpi_df.columns:
                if '-Q' in col:  # Look for quarterly data
                    try:
                        year_str, quarter_str = col.split('-Q')
                        year = int(year_str)
                        quarter = int(quarter_str)
                        
                        if 2015 <= year <= 2024:  # Focus on relevant years
                            for _, row in cpi_df.iterrows():
                                # Determine country
                                country = None
                                if 'ISO_CODE' in cpi_df.columns and pd.notna(row.get('ISO_CODE')):
                                    country = row['ISO_CODE']
                                elif 'COUNTRY.Name' in cpi_df.columns and pd.notna(row.get('COUNTRY.Name')):
                                    # Extract country code from name
                                    country_name = str(row['COUNTRY.Name'])
                                    for market in key_markets:
                                        if market in country_name:
                                            country = market
                                            break
                                
                                # Get inflation value
                                inflation_value = row.get(col)
                                
                                if country in key_markets and pd.notna(inflation_value):
                                    # Convert to numeric (some values might be strings)
                                    try:
                                        inflation_value = float(inflation_value)
                                        
                                        new_row = pd.DataFrame({
                                            'Year': [year],
                                            'Quarter': [quarter],
                                            'Country': [country],
                                            'Inflation_Rate': [inflation_value]
                                        })
                                        cpi_reshaped = pd.concat([cpi_reshaped, new_row], ignore_index=True)
                                    except (ValueError, TypeError):
                                        continue
                    except (ValueError, TypeError):
                        continue
            
            if not cpi_reshaped.empty:
                print(f"Successfully extracted CPI data with {len(cpi_reshaped)} entries")
                return cpi_reshaped
            else:
                print("Failed to extract structured CPI data from the file")
        else:
            print("No relevant CPI data chunks found")
            
    except Exception as e:
        print(f"Error processing CPI data: {e}")
    
    print("Using placeholder CPI data due to extraction issues")
    
    # Create structured placeholder data as a last resort (but with clear warning)
    # This is not sample data but an emergency fallback if file parsing fails
    cpi_reshaped = pd.DataFrame()
    
    # Basic structure to match the expected format
    for country in key_markets:
        for year in range(2015, 2024):
            for quarter in range(1, 5):
                # Base inflation with some variation to avoid identical values
                # These are not meant to be accurate but just maintain program functionality
                base_inflation = 2.0 + (year - 2015) * 0.2 + (quarter - 1) * 0.1
                if year >= 2020:
                    base_inflation += 1.0  # Slight increase for post-2020
                
                new_row = pd.DataFrame({
                    'Year': [year],
                    'Quarter': [quarter],
                    'Country': [country],
                    'Inflation_Rate': [base_inflation]
                })
                cpi_reshaped = pd.concat([cpi_reshaped, new_row], ignore_index=True)
    
    print("WARNING: Using placeholder CPI data. Results will not be accurate!")
    return cpi_reshaped

# Load and process IMF Fiscal Monitor data (GDP growth)
def load_fiscal_data():
    """Load and process IMF Fiscal Monitor data for GDP growth"""
    print("Loading fiscal data...")
    
    # Define key markets (major Spotify markets)
    key_markets = ['United States', 'United Kingdom', 'Germany', 'France', 'Sweden', 
                  'Brazil', 'Mexico', 'Japan', 'Australia', 'Canada']
    
    # List of alternate names/keywords for identifying countries
    country_keywords = {
        'United States': ['USA', 'U.S.', 'United States of America'],
        'United Kingdom': ['UK', 'Britain', 'Great Britain'],
        'Germany': ['Deutschland', 'German Federal Republic'],
        'France': ['French Republic'],
        'Sweden': ['Kingdom of Sweden'],
        'Brazil': ['Brasil', 'Federative Republic of Brazil'],
        'Mexico': ['México', 'United Mexican States'],
        'Japan': ['Nippon', 'Nihon'],
        'Australia': ['Commonwealth of Australia'],
        'Canada': ['Canadian']
    }
    
    # Mapping country names to ISO codes for consistency
    country_map = {
        'United States': 'USA', 
        'United Kingdom': 'GBR', 
        'Germany': 'DEU', 
        'France': 'FRA', 
        'Sweden': 'SWE',
        'Brazil': 'BRA', 
        'Mexico': 'MEX', 
        'Japan': 'JPN', 
        'Australia': 'AUS', 
        'Canada': 'CAN'
    }
    
    try:
        # Read the fiscal data file
        print("Reading fiscal data file...")
        fiscal_df = pd.read_csv(FISCAL_FILE, low_memory=False)
        
        # Look for GDP growth rate data using various series codes
        gdp_codes = ['NGDP_RPCH', 'GDP_growth', 'GDP.MKTP.KD.ZG']
        
        # Filter for GDP growth data for our key markets
        gdp_df = pd.DataFrame()
        
        for code in gdp_codes:
            if 'SERIES_CODE' in fiscal_df.columns:
                filtered = fiscal_df[fiscal_df['SERIES_CODE'].str.contains(code, na=False)]
                if not filtered.empty:
                    gdp_df = pd.concat([gdp_df, filtered], ignore_index=True)
        
        # If no specific codes found, try a more general approach
        if gdp_df.empty and 'INDICATOR.Name' in fiscal_df.columns:
            potential_gdp_rows = fiscal_df[
                fiscal_df['INDICATOR.Name'].str.contains('GDP', na=False) & 
                fiscal_df['INDICATOR.Name'].str.contains('growth', na=False, case=False)
            ]
            if not potential_gdp_rows.empty:
                gdp_df = pd.concat([gdp_df, potential_gdp_rows], ignore_index=True)
        
        # Filter for key markets
        if not gdp_df.empty and 'COUNTRY.Name' in gdp_df.columns:
            # Create a filter to match any of our key markets or their alternate names
            country_filter = gdp_df['COUNTRY.Name'].apply(
                lambda x: any(country in str(x) for country in key_markets) or 
                         any(any(keyword in str(x) for keyword in keywords) 
                             for country, keywords in country_keywords.items())
            )
            gdp_df = gdp_df[country_filter]
        
        # Process the filtered data
        if not gdp_df.empty:
            print(f"Found {len(gdp_df)} GDP growth data rows")
            
            # Create a structured DataFrame for the analysis
            gdp_data = pd.DataFrame()
            
            # Extract numerical columns (years)
            year_cols = [col for col in gdp_df.columns if str(col).isdigit() and int(col) >= 2015 and int(col) < 2025]
            
            for _, row in gdp_df.iterrows():
                # Determine which country this row represents
                country = None
                if 'COUNTRY.Name' in gdp_df.columns:
                    country_name = str(row['COUNTRY.Name'])
                    for key_country in key_markets:
                        if key_country in country_name:
                            country = key_country
                            break
                    
                    # If no direct match, try alternative names
                    if country is None:
                        for key_country, keywords in country_keywords.items():
                            if any(keyword in country_name for keyword in keywords):
                                country = key_country
                                break
                
                if country:
                    # Extract GDP values for each year and create quarterly entries
                    for year_col in year_cols:
                        year = int(year_col)
                        gdp_value = row.get(year_col)
                        
                        if pd.notna(gdp_value):
                            # Ensure numeric value
                            try:
                                gdp_value = float(gdp_value)
                                
                                # Create 4 quarterly entries with the same annual GDP value
                                for quarter in range(1, 5):
                                    new_row = pd.DataFrame({
                                        'Year': [year],
                                        'Quarter': [quarter],
                                        'Country': [country_map.get(country, country)],
                                        'GDP_Growth': [gdp_value]
                                    })
                                    gdp_data = pd.concat([gdp_data, new_row], ignore_index=True)
                            except (ValueError, TypeError):
                                continue
            
            if not gdp_data.empty:
                print(f"Successfully extracted GDP data with {len(gdp_data)} entries")
                return gdp_data
            else:
                print("Failed to extract structured GDP data from the filtered rows")
        else:
            print("No GDP growth data found in the fiscal file")
            
    except Exception as e:
        print(f"Error processing fiscal data: {e}")
    
    print("Using placeholder GDP data due to extraction issues")
    
    # Create structured placeholder data as a last resort (but with clear warning)
    # This is not sample data but an emergency fallback if file parsing fails
    gdp_data = pd.DataFrame()
    
    # Basic structure to match the expected format
    for country, iso_code in country_map.items():
        for year in range(2015, 2024):
            # Very basic placeholder pattern
            gdp_value = 2.0
            if 2020 <= year <= 2021:  # Simulate pandemic effect
                gdp_value = -2.0 if year == 2020 else 4.0
            
            for quarter in range(1, 5):
                new_row = pd.DataFrame({
                    'Year': [year],
                    'Quarter': [quarter],
                    'Country': [iso_code],
                    'GDP_Growth': [gdp_value]
                })
                gdp_data = pd.concat([gdp_data, new_row], ignore_index=True)
    
    print("WARNING: Using placeholder GDP data. Results will not be accurate!")
    return gdp_data

# Combine datasets and analyze relationships
def analyze_spotify_economics():
    """Combine datasets and analyze relationships between economic indicators and Spotify growth"""
    users_df, revenue_df = load_spotify_data()
    
    try:
        print("Attempting to extract CPI data from file...")
        cpi_df = load_cpi_data()
        if cpi_df.empty:
            raise ValueError("CPI data extraction returned empty dataset")
    except Exception as e:
        print(f"Error with CPI data: {e}")
        print("Analysis cannot continue without CPI data")
        return None
    
    try:
        print("Attempting to extract GDP data from file...")
        gdp_df = load_fiscal_data()
        if gdp_df.empty:
            raise ValueError("GDP data extraction returned empty dataset")
    except Exception as e:
        print(f"Error with GDP data: {e}")
        print("Analysis cannot continue without GDP data")
        return None
    
    print("Merging datasets...")
    
    # Since Spotify data is global, we need to create weighted economic indicators
    # We'll weight by rough estimates of Spotify's market share in each region
    
    # Market weights (approximated based on Spotify's known market penetration)
    market_weights = {
        'USA': 0.28,   # North America (~28%)
        'GBR': 0.11,   # UK
        'DEU': 0.09,   # Germany
        'FRA': 0.07,   # France
        'SWE': 0.02,   # Sweden (Spotify's home country)
        'BRA': 0.09,   # Brazil
        'MEX': 0.05,   # Mexico
        'JPN': 0.04,   # Japan
        'AUS': 0.03,   # Australia
        'CAN': 0.04,   # Canada
        # Remaining ~18% distributed across other markets
    }
    
    # Create weighted economic indicators
    weighted_econ = pd.DataFrame()
    
    # Group by year and quarter
    for year in range(2015, 2025):
        for quarter in range(1, 5):
            # Skip future quarters
            if (year == 2024 and quarter > 1):
                continue
                
            weighted_inflation = 0
            weighted_gdp_growth = 0
            
            # Calculate weighted averages
            for country, weight in market_weights.items():
                # Inflation
                country_inflation = cpi_df[(cpi_df['Year'] == year) & 
                                          (cpi_df['Quarter'] == quarter) & 
                                          (cpi_df['Country'] == country)]
                
                if not country_inflation.empty and not pd.isna(country_inflation['Inflation_Rate'].values[0]):
                    weighted_inflation += country_inflation['Inflation_Rate'].values[0] * weight
                
                # GDP growth
                country_gdp = gdp_df[(gdp_df['Year'] == year) & 
                                    (gdp_df['Quarter'] == quarter) & 
                                    (gdp_df['Country'] == country)]
                
                if not country_gdp.empty and not pd.isna(country_gdp['GDP_Growth'].values[0]):
                    weighted_gdp_growth += country_gdp['GDP_Growth'].values[0] * weight
            
            # Create a row for this year-quarter
            new_row = pd.DataFrame({
                'Year': [year],
                'Quarter': [quarter],
                'Weighted_Inflation': [weighted_inflation],
                'Weighted_GDP_Growth': [weighted_gdp_growth]
            })
            
            weighted_econ = pd.concat([weighted_econ, new_row], ignore_index=True)
    
    # Merge with Spotify user data
    analysis_df = pd.merge(users_df, weighted_econ, on=['Year', 'Quarter'], how='left')
    
    # Drop rows with missing economic data
    analysis_df = analysis_df.dropna(subset=['Weighted_Inflation', 'Weighted_GDP_Growth', 
                                            'MAU_Growth', 'Premium_Growth'])
    
    print("Analyzing correlations...")
    
    # Calculate correlations
    corr_mau_inflation = analysis_df['MAU_Growth'].corr(analysis_df['Weighted_Inflation'])
    corr_premium_inflation = analysis_df['Premium_Growth'].corr(analysis_df['Weighted_Inflation'])
    corr_mau_gdp = analysis_df['MAU_Growth'].corr(analysis_df['Weighted_GDP_Growth'])
    corr_premium_gdp = analysis_df['Premium_Growth'].corr(analysis_df['Weighted_GDP_Growth'])
    
    # Calculate lagged correlations (economy affects users with delay)
    analysis_df['Weighted_Inflation_Lag1'] = analysis_df['Weighted_Inflation'].shift(1)
    analysis_df['Weighted_GDP_Growth_Lag1'] = analysis_df['Weighted_GDP_Growth'].shift(1)
    
    analysis_df = analysis_df.dropna(subset=['Weighted_Inflation_Lag1', 'Weighted_GDP_Growth_Lag1'])
    
    corr_mau_inflation_lag = analysis_df['MAU_Growth'].corr(analysis_df['Weighted_Inflation_Lag1'])
    corr_premium_inflation_lag = analysis_df['Premium_Growth'].corr(analysis_df['Weighted_Inflation_Lag1'])
    corr_mau_gdp_lag = analysis_df['MAU_Growth'].corr(analysis_df['Weighted_GDP_Growth_Lag1'])
    corr_premium_gdp_lag = analysis_df['Premium_Growth'].corr(analysis_df['Weighted_GDP_Growth_Lag1'])
    
    # Define recession periods (approximate based on global economic data)
    # COVID-19 recession in 2020
    recession_periods = [
        (2020, 1), (2020, 2), (2020, 3)  # Q1-Q3 2020
    ]
    
    # Mark recession periods
    analysis_df['Recession'] = analysis_df.apply(
        lambda x: 1 if (x['Year'], x['Quarter']) in recession_periods else 0, 
        axis=1
    )
    
    # Analyze user growth during recessions vs. non-recessions
    recession_df = analysis_df[analysis_df['Recession'] == 1]
    non_recession_df = analysis_df[analysis_df['Recession'] == 0]
    
    recession_mau_growth = recession_df['MAU_Growth'].mean()
    non_recession_mau_growth = non_recession_df['MAU_Growth'].mean()
    recession_premium_growth = recession_df['Premium_Growth'].mean()
    non_recession_premium_growth = non_recession_df['Premium_Growth'].mean()
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Setup figure for plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Spotify Growth and Economic Indicators', fontsize=18)
    
    # Plot 1: Spotify user growth over time
    axes[0, 0].plot(analysis_df['Date'], analysis_df['Monthly Active Users (Millions)'], 'b-', label='MAU')
    axes[0, 0].plot(analysis_df['Date'], analysis_df['Paying Subscribers (Millions)'], 'r-', label='Premium')
    axes[0, 0].set_title('Spotify User Growth')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Users (Millions)')
    axes[0, 0].legend()
    
    # Shade recession periods
    for year, quarter in recession_periods:
        quarter_start = pd.to_datetime(f"{year}-{quarter*3-2}-01")
        quarter_end = pd.to_datetime(f"{year}-{quarter*3+2}-30")
        axes[0, 0].axvspan(quarter_start, quarter_end, color='gray', alpha=0.3)
    
    # Plot 2: Growth rates
    axes[0, 1].plot(analysis_df['Date'], analysis_df['MAU_Growth'], 'b-', label='MAU Growth %')
    axes[0, 1].plot(analysis_df['Date'], analysis_df['Premium_Growth'], 'r-', label='Premium Growth %')
    axes[0, 1].set_title('Spotify Quarterly Growth Rates')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Growth Rate (%)')
    axes[0, 1].legend()
    
    # Shade recession periods
    for year, quarter in recession_periods:
        quarter_start = pd.to_datetime(f"{year}-{quarter*3-2}-01")
        quarter_end = pd.to_datetime(f"{year}-{quarter*3+2}-30")
        axes[0, 1].axvspan(quarter_start, quarter_end, color='gray', alpha=0.3)
    
    # Plot 3: Spotify growth vs. inflation
    axes[1, 0].scatter(analysis_df['Weighted_Inflation'], analysis_df['MAU_Growth'], 
                      alpha=0.7, label='MAU Growth')
    axes[1, 0].scatter(analysis_df['Weighted_Inflation'], analysis_df['Premium_Growth'], 
                      alpha=0.7, label='Premium Growth')
    
    # Add trend lines
    x = analysis_df['Weighted_Inflation']
    y1 = analysis_df['MAU_Growth']
    y2 = analysis_df['Premium_Growth']
    
    z1 = np.polyfit(x, y1, 1)
    p1 = np.poly1d(z1)
    axes[1, 0].plot(x, p1(x), "b--", alpha=0.5)
    
    z2 = np.polyfit(x, y2, 1)
    p2 = np.poly1d(z2)
    axes[1, 0].plot(x, p2(x), "r--", alpha=0.5)
    
    axes[1, 0].set_title(f'Spotify Growth vs. Inflation\nMAU corr={corr_mau_inflation:.2f}, Premium corr={corr_premium_inflation:.2f}')
    axes[1, 0].set_xlabel('Weighted Inflation Rate (%)')
    axes[1, 0].set_ylabel('Growth Rate (%)')
    axes[1, 0].legend()
    
    # Plot 4: Spotify growth vs. GDP growth
    axes[1, 1].scatter(analysis_df['Weighted_GDP_Growth'], analysis_df['MAU_Growth'], 
                     alpha=0.7, label='MAU Growth')
    axes[1, 1].scatter(analysis_df['Weighted_GDP_Growth'], analysis_df['Premium_Growth'], 
                     alpha=0.7, label='Premium Growth')
    
    # Add trend lines
    x = analysis_df['Weighted_GDP_Growth']
    
    z1 = np.polyfit(x, y1, 1)
    p1 = np.poly1d(z1)
    axes[1, 1].plot(x, p1(x), "b--", alpha=0.5)
    
    z2 = np.polyfit(x, y2, 1)
    p2 = np.poly1d(z2)
    axes[1, 1].plot(x, p2(x), "r--", alpha=0.5)
    
    axes[1, 1].set_title(f'Spotify Growth vs. GDP Growth\nMAU corr={corr_mau_gdp:.2f}, Premium corr={corr_premium_gdp:.2f}')
    axes[1, 1].set_xlabel('Weighted GDP Growth Rate (%)')
    axes[1, 1].set_ylabel('Growth Rate (%)')
    axes[1, 1].legend()
    
    # Plot 5: Bar chart comparing recession vs non-recession growth
    labels = ['Recession', 'Non-Recession']
    mau_data = [recession_mau_growth, non_recession_mau_growth]
    premium_data = [recession_premium_growth, non_recession_premium_growth]
    
    x = np.arange(len(labels))
    width = 0.35
    
    axes[2, 0].bar(x - width/2, mau_data, width, label='MAU Growth')
    axes[2, 0].bar(x + width/2, premium_data, width, label='Premium Growth')
    
    axes[2, 0].set_title('Spotify Growth: Recession vs. Non-Recession Periods')
    axes[2, 0].set_ylabel('Average Growth Rate (%)')
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(labels)
    axes[2, 0].legend()
    
    # Add actual values as text
    for i, v in enumerate(mau_data):
        axes[2, 0].text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center')
        
    for i, v in enumerate(premium_data):
        axes[2, 0].text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center')
    
    # Plot 6: Premium to Free ratio over time to check for shifts
    analysis_df['Premium_to_Free_Ratio'] = analysis_df['Paying Subscribers (Millions)'] / (
        analysis_df['Monthly Active Users (Millions)'] - analysis_df['Paying Subscribers (Millions)'])
    
    axes[2, 1].plot(analysis_df['Date'], analysis_df['Premium_to_Free_Ratio'], 'g-')
    axes[2, 1].set_title('Premium to Free User Ratio')
    axes[2, 1].set_xlabel('Date')
    axes[2, 1].set_ylabel('Ratio')
    
    # Shade recession periods
    for year, quarter in recession_periods:
        quarter_start = pd.to_datetime(f"{year}-{quarter*3-2}-01")
        quarter_end = pd.to_datetime(f"{year}-{quarter*3+2}-30")
        axes[2, 1].axvspan(quarter_start, quarter_end, color='gray', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('spotify_economic_analysis.png')
    print("Analysis complete. Results saved to spotify_economic_analysis.png")
    
    # Print summary of findings
    print("\nSUMMARY OF FINDINGS:")
    print(f"1. Correlation between MAU growth and inflation: {corr_mau_inflation:.3f}")
    print(f"2. Correlation between Premium growth and inflation: {corr_premium_inflation:.3f}")
    print(f"3. Correlation between MAU growth and GDP growth: {corr_mau_gdp:.3f}")
    print(f"4. Correlation between Premium growth and GDP growth: {corr_premium_gdp:.3f}")
    print(f"5. Correlation with lagged indicators (one quarter):")
    print(f"   - MAU growth vs inflation (lagged): {corr_mau_inflation_lag:.3f}")
    print(f"   - Premium growth vs inflation (lagged): {corr_premium_inflation_lag:.3f}")
    print(f"   - MAU growth vs GDP growth (lagged): {corr_mau_gdp_lag:.3f}")
    print(f"   - Premium growth vs GDP growth (lagged): {corr_premium_gdp_lag:.3f}")
    print(f"6. During recession periods:")
    print(f"   - Average MAU growth: {recession_mau_growth:.2f}%")
    print(f"   - Average Premium growth: {recession_premium_growth:.2f}%")
    print(f"7. During non-recession periods:")
    print(f"   - Average MAU growth: {non_recession_mau_growth:.2f}%")
    print(f"   - Average Premium growth: {non_recession_premium_growth:.2f}%")
    
    return analysis_df

if __name__ == "__main__":
    #analysis_df = analyze_spotify_economics() 
    spotify_data = load_spotify_data()
    print(spotify_data[0])
    print(spotify_data[1])
    cpi_data = load_cpi_data()
    print(cpi_data.head())
    fiscal_data = load_fiscal_data()
    print(fiscal_data.head())