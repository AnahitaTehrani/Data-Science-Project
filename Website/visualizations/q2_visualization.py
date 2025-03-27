import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

def load_and_preprocess_data(file_path='data/daily_data_with_lags.csv'):
    """
    Load and preprocess the sentiment and stock data
    
    Parameters:
    file_path (str): Path to the data file
    
    Returns:
    tuple: (complete_dataframe, filtered_dataframe)
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Ensure date column is datetime if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # Set date as index if it exists
        df.set_index('date', inplace=True)
    
    # Filter the dataframe to include only rows with valid data
    # Make sure to only include trading days with non-null data for key columns
    df_filtered = df.dropna(subset=['combined_sentiment', 'next_day_return', 'news_count'])
    df_filtered = df_filtered[df_filtered['is_trading_day']]
    
    return df, df_filtered

def create_sentiment_time_series(df):
    """
    Create a time series plot of stock price and sentiment over time
    
    Parameters:
    df (pd.DataFrame): Dataframe with stock and sentiment data
    
    Returns:
    go.Figure: Plotly figure object with the time series
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add close price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close'],
            name="Close Price",
            line=dict(color="#1A73E8", width=2)
        ),
        secondary_y=False,
    )
    
    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['combined_sentiment'],
            name="Sentiment Score",
            line=dict(color="#1DB954", width=2)
        ),
        secondary_y=True,
    )
    
    # Set x-axis title
    fig.update_xaxes(title_text="Date")
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Stock Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
    
    # Add title and update layout
    fig.update_layout(
        title="Spotify Stock Price and News Sentiment Over Time",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    return fig

def create_sentiment_return_scatter(df_filtered, time_period='next_day'):
    """
    Create a scatter plot of sentiment vs returns
    
    Parameters:
    df_filtered (pd.DataFrame): Filtered dataframe with sentiment and return data
    time_period (str): 'next_day' or 'same_day' to choose between next_day_return and daily_return
    
    Returns:
    go.Figure: Plotly figure object with the scatter plot
    """
    # Choose the return column based on time_period
    return_col = 'next_day_return' if time_period == 'next_day' else 'daily_return'
    return_label = 'Next-Day Return (%)' if time_period == 'next_day' else 'Same-Day Return (%)'
    title_prefix = 'Next-Day' if time_period == 'next_day' else 'Same-Day'
    
    # Calculate trendline
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_filtered['combined_sentiment'],
        df_filtered[return_col]
    )
    
    # Create figure
    fig = px.scatter(
        df_filtered,
        x='combined_sentiment',
        y=return_col,
        size='news_count',
        color='news_count',
        color_continuous_scale='viridis',
        opacity=0.7,
        labels={
            'combined_sentiment': 'News Sentiment Score',
            return_col: return_label,
            'news_count': 'Number of News Articles'
        },
        title=f"Spotify News Sentiment Impact on {title_prefix} Stock Returns",
        hover_data={
            'combined_sentiment': ':.3f',
            return_col: ':.2f%',
            'news_count': True
        }
    )
    
    # Add trendline
    x_range = np.linspace(df_filtered['combined_sentiment'].min(), df_filtered['combined_sentiment'].max(), 100)
    y_range = slope * x_range + intercept
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name=f'Trend (r={r_value:.3f})'
        )
    )
    
    # Add reference lines at zero
    fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="gray", opacity=0.3)
    fig.add_vline(x=0, line_width=1, line_dash="solid", line_color="gray", opacity=0.3)
    
    # Add quadrant labels
    max_y = df_filtered[return_col].max() * 0.9
    max_x = df_filtered['combined_sentiment'].max() * 0.9
    min_y = df_filtered[return_col].min() * 0.9
    min_x = df_filtered['combined_sentiment'].min() * 0.9
    
    fig.add_annotation(
        x=max_x*0.7, y=max_y*0.7,
        text="Positive Sentiment<br>Positive Return",
        showarrow=False,
        bgcolor="white",
        opacity=0.8,
        bordercolor="black",
        borderwidth=1
    )
    fig.add_annotation(
        x=min_x*0.7, y=max_y*0.7,
        text="Negative Sentiment<br>Positive Return",
        showarrow=False,
        bgcolor="white",
        opacity=0.8,
        bordercolor="black",
        borderwidth=1
    )
    fig.add_annotation(
        x=max_x*0.7, y=min_y*0.7,
        text="Positive Sentiment<br>Negative Return",
        showarrow=False,
        bgcolor="white",
        opacity=0.8,
        bordercolor="black",
        borderwidth=1
    )
    fig.add_annotation(
        x=min_x*0.7, y=min_y*0.7,
        text="Negative Sentiment<br>Negative Return",
        showarrow=False,
        bgcolor="white",
        opacity=0.8,
        bordercolor="black",
        borderwidth=1
    )
    
    # Update layout for aesthetics
    fig.update_layout(
        template="plotly_white",
        legend_title="Number of News Articles",
        xaxis_title="News Sentiment Score",
        yaxis_title=return_label,
        yaxis_tickformat='.1f%',  # Format y-axis as percentage
        hovermode="closest",
        margin=dict(l=60, r=30, t=80, b=60),
        # Fix legend positioning
        legend=dict(
            orientation="h",     # Horizontal legend
            yanchor="top",       # Anchor top of legend to specified y position
            y=-0.15,             # Position legend below the plot
            xanchor="center",    # Anchor center of legend to specified x position
            x=0.5               # Center legend horizontally
        ),
        # Adjust colorbar positioning
        coloraxis=dict(
            colorbar=dict(
                title="Numb. of News Art.",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05,
                len=1
            )
        )
    )
    
    return fig

def create_visualization(df_filtered, viz_type='sentiment_returns', time_period='next_day'):
    """
    Create the specified visualization
    
    Parameters:
    df_filtered (pd.DataFrame): Filtered dataframe with sentiment and stock data
    viz_type (str): Type of visualization to create
    time_period (str): 'next_day' or 'same_day' (for scatter plots)
    
    Returns:
    go.Figure: Plotly figure object with the requested visualization
    """
    if viz_type == 'sentiment_returns':
        return create_sentiment_return_scatter(df_filtered, time_period)
    elif viz_type == 'time_series':
        return create_sentiment_time_series(df_filtered)
    else:
        # Default to sentiment vs returns
        return create_sentiment_return_scatter(df_filtered, time_period)