import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_user_growth_visualization(data, selected_year='all'):
    """
    Create a visualization for Spotify's user growth over time.
    
    Args:
        data (pd.DataFrame): DataFrame containing user growth data
        selected_year (str or int): Selected year to filter data, or 'all' for all years
    
    Returns:
        plotly.graph_objects.Figure: The visualization figure
    """
    # Make a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Convert quarterly dates (e.g., "Q1 2015") to datetime format
    def parse_quarter_date(quarter_str):
        parts = quarter_str.split()
        quarter = int(parts[0][1:])
        year = int(parts[1])
        month = 3 * quarter - 2  # Q1->1, Q2->4, Q3->7, Q4->10
        return pd.Timestamp(year=year, month=month, day=15)
    
    # Apply the custom date parser and sort by date
    df['Date'] = df['Date'].apply(parse_quarter_date)
    df = df.sort_values('Date')
    
    # Extract year for filtering
    df['Year'] = df['Date'].dt.year
    
    # Calculate percentage of paying subscribers
    df['Percentage Paying'] = (df['Paying Subscribers (Millions)'] / df['Monthly Active Users (Millions)']) * 100
    
    # Filter by year if specified
    if selected_year != 'all':
        selected_year = int(selected_year)
        df = df[df['Year'] == selected_year]
    
    # Create a figure with two subplots (one for user counts, one for percentage)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Spotify User Growth", "Percentage of Paying Users")
    )
    
    # Add traces for Monthly Active Users
    fig.add_trace(
        go.Scatter(
            x=df['Date'], 
            y=df['Monthly Active Users (Millions)'],
            mode='lines+markers',
            name='Monthly Active Users',
            line=dict(color='#1DB954', width=3),  # Spotify green
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Add traces for Paying Subscribers
    fig.add_trace(
        go.Scatter(
            x=df['Date'], 
            y=df['Paying Subscribers (Millions)'],
            mode='lines+markers',
            name='Paying Subscribers',
            line=dict(color='#191414', width=3),  # Spotify black
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Add trace for Percentage of Paying Users
    fig.add_trace(
        go.Scatter(
            x=df['Date'], 
            y=df['Percentage Paying'],
            mode='lines+markers',
            name='% Paying Users',
            line=dict(color='#FF7C00', width=2),  # Orange
            marker=dict(size=6)
        ),
        row=2, col=1
    )
    
    # Calculate key metrics
    if len(df) > 1:
        first = df.iloc[0]
        latest = df.iloc[-1]
        growth_mau = ((latest['Monthly Active Users (Millions)'] - first['Monthly Active Users (Millions)']) / 
                     first['Monthly Active Users (Millions)']) * 100
        growth_paying = ((latest['Paying Subscribers (Millions)'] - first['Paying Subscribers (Millions)']) / 
                        first['Paying Subscribers (Millions)']) * 100
        
        # Calculate time difference in years for annual growth rate
        time_diff = (latest['Date'] - first['Date']).days / 365.25
        if time_diff > 0:
            annual_mau_growth = ((latest['Monthly Active Users (Millions)'] / first['Monthly Active Users (Millions)']) ** (1/time_diff) - 1) * 100
            annual_sub_growth = ((latest['Paying Subscribers (Millions)'] / first['Paying Subscribers (Millions)']) ** (1/time_diff) - 1) * 100
        else:
            annual_mau_growth = 0
            annual_sub_growth = 0
            
        # Add annotations for latest values
        fig.add_annotation(
            x=latest['Date'],
            y=latest['Monthly Active Users (Millions)'],
            text=f"{int(latest['Monthly Active Users (Millions)'])}M",
            showarrow=True,
            arrowhead=1,
            row=1, col=1
        )
        
        fig.add_annotation(
            x=latest['Date'],
            y=latest['Paying Subscribers (Millions)'],
            text=f"{int(latest['Paying Subscribers (Millions)'])}M",
            showarrow=True,
            arrowhead=1,
            row=1, col=1
        )
        
        # Add a text box with key metrics
        stats_text = (
            f"<b>Growth Summary:</b><br>"
            f"• Monthly Active Users: {growth_mau:.1f}%<br>"
            f"• Paying Subscribers: {growth_paying:.1f}%<br>"
            f"• Current Paying Ratio: {latest['Percentage Paying']:.1f}%"
        )
        
        if time_diff > 0:
            stats_text += (
                f"<br>• Avg. Annual Growth (MAU): {annual_mau_growth:.1f}%<br>"
                f"• Avg. Annual Growth (Subs): {annual_sub_growth:.1f}%"
            )
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            bgcolor="white",
            opacity=0.8,
            bordercolor="lightgrey",
            borderwidth=1,
            borderpad=4,
            align="left",
            font=dict(size=12)
        )
    
    # Update layout
    fig.update_layout(
        height=700,
        template='plotly', 
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=70, b=50),
        hovermode="x unified"
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Users (Millions)", row=1, col=1)
    fig.update_yaxes(title_text="% Paying Users", row=2, col=1)
    
    # Update x-axis
    fig.update_xaxes(
        title_text="Date",
        tickformat="%Y-%b",
        row=2, col=1
    )
    
    return fig

def get_research_question_layout(df, selected_year):
    """
    This function is no longer needed as we're handling layout in app.py now.
    Keeping it here just for backward compatibility.
    """
    fig = create_user_growth_visualization(df, selected_year)
    return fig