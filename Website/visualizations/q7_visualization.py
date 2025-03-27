# q7_visualization.py
# Visualization for Research Question 7: How do price changes affect Spotify's growth?

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def process_price_data(price_df):
    """Process the price data to create date ranges."""
    price_periods = []
    
    for _, row in price_df.iterrows():
        period_str = row['Period']
        price = row['Price']
        
        # Skip rows with missing data
        if pd.isna(period_str) or pd.isna(price):
            continue
            
        # Parse different period formats
        if "–" in period_str or "-" in period_str:
            # Handle date ranges
            parts = period_str.replace("–", "-").split("-")
            start_str = parts[0].strip()
            end_str = parts[1].strip() if len(parts) > 1 else None
            
            # Parse start date
            try:
                if len(start_str) == 4:  # Year only
                    start_date = datetime.strptime(start_str, "%Y")
                else:  # Month and year
                    start_date = datetime.strptime(start_str, "%B %Y")
            except:
                continue
                
            # Parse end date
            if end_str:
                try:
                    if len(end_str) == 4:  # Year only
                        end_date = datetime.strptime(end_str, "%Y")
                        # Add a year to include the whole year
                        end_date = datetime(end_date.year + 1, 1, 1)
                    else:  # Month and year
                        end_date = datetime.strptime(end_str, "%B %Y")
                        # Add a month to include the whole month
                        if end_date.month == 12:
                            end_date = datetime(end_date.year + 1, 1, 1)
                        else:
                            end_date = datetime(end_date.year, end_date.month + 1, 1)
                except:
                    continue
            else:
                continue
                
        elif "Since" in period_str:
            # Handle "Since <date>" format
            try:
                date_str = period_str.replace("Since", "").strip()
                start_date = datetime.strptime(date_str, "%B %Y")
                end_date = datetime.now()  # Current date as end date
            except:
                continue
        else:
            # Skip unsupported formats
            continue
            
        price_periods.append((start_date, end_date, price))
    
    return price_periods

def process_premium_data(premium_df):
    """Process the premium user data to create quarterly data."""
    premium_data = []
    
    for _, row in premium_df.iterrows():
        # Skip header rows or rows with missing data
        if not isinstance(row[0], str) or ";" not in row[0]:
            continue
            
        # Parse the semicolon-separated string
        parts = row[0].split(";")
        if len(parts) < 3:
            continue
            
        quarter = parts[1].strip()
        users_str = parts[2].strip()
        
        # Extract quarter and year
        try:
            q, y = quarter.split()
            year = int(y)
            month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}[q]
            date = datetime(year, month, 1)
            users = int(users_str)
            premium_data.append((date, users))
        except:
            continue
    
    return premium_data

def match_price_to_premium(premium_data, price_periods):
    """Match each premium data point with the corresponding price."""
    final_data = []
    
    for date, users in premium_data:
        # Find the matching price period
        matching_price = None
        for start, end, price in price_periods:
            if start <= date <= end:
                matching_price = price
                break
                
        if matching_price is not None:
            final_data.append({
                'Date': date,
                'PremiumUsers_Mio': users,
                'Price_EUR': matching_price
            })
    
    # Create DataFrame from the matched data
    final_df = pd.DataFrame(final_data)
    
    return final_df

def create_scatter_visualization(final_df, correlation):
    """Create a scatter plot of price vs premium users."""
    fig = px.scatter(
        final_df, 
        x="Price_EUR", 
        y="PremiumUsers_Mio",
        title=f"Correlation Between Spotify Price and Number of Premium Users (r = {correlation:.2f})",
        labels={
            "Price_EUR": "Price in EUR",
            "PremiumUsers_Mio": "Premium Users (in millions)"
        },
        template="plotly_white"
    )
    
    # Update marker size
    fig.update_traces(marker=dict(size=12, color="#1DB954"))  # Spotify green
    
    # Add trendline
    trendline = px.scatter(
        final_df, 
        x="Price_EUR", 
        y="PremiumUsers_Mio", 
        trendline="ols"
    ).data[1]
    
    fig.add_trace(trendline)
    
    # Customize layout
    fig.update_layout(
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
        plot_bgcolor="white",
        height=550,
    )
    
    return fig

def create_timeseries_visualization(final_df):
    """Create a time series chart of price and premium users over time."""
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=final_df["Date"],
            y=final_df["PremiumUsers_Mio"],
            name="Premium Users",
            line=dict(color="#1DB954", width=3)  # Spotify green
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=final_df["Date"],
            y=final_df["Price_EUR"],
            name="Price",
            line=dict(color="#E21B3C", width=3)  # Red
        ),
        secondary_y=True,
    )
    
    # Add figure title
    fig.update_layout(
        title="Spotify: Development of Price and Number of Premium Users",
        template="plotly_white",
        height=550,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text="Year")
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Premium Users (in millions)", secondary_y=False)
    fig.update_yaxes(title_text="Price (EUR)", secondary_y=True)
    
    return fig

def create_price_impact_visualization(price_df, premium_df, viz_type='scatter'):
    """Create the visualization for Question 7.
    
    Parameters:
    price_df (DataFrame): DataFrame containing price data
    premium_df (DataFrame): DataFrame containing premium user data
    viz_type (str): The type of visualization to create ('scatter' or 'timeseries')
    
    Returns:
    fig: A plotly figure object
    """
    try:
        # Process the price data to create date ranges
        price_periods = process_price_data(price_df)
        
        # Process the premium user data
        premium_data = process_premium_data(premium_df)
        
        # Match premium data with price periods
        final_df = match_price_to_premium(premium_data, price_periods)
        
        # Calculate correlation
        correlation = final_df["Price_EUR"].corr(final_df["PremiumUsers_Mio"])
        
        # Create the visualization based on the selected type
        if viz_type == 'scatter':
            return create_scatter_visualization(final_df, correlation)
        else:  # timeseries
            return create_timeseries_visualization(final_df)
    except Exception as e:
        # Return empty figure with error message
        return {
            'data': [],
            'layout': {
                'title': f"Error processing data: {str(e)}",
                'height': 500
            }
        }