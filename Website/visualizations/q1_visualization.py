import plotly.express as px
import pandas as pd

def create_user_growth_visualization(data, selected_year):
    """
    Create a visualization for Spotify's user growth over time.
    
    Args:
        data (pd.DataFrame): DataFrame containing user growth data
        selected_year (int or str): Selected year to filter data, or 'all' for all years
    
    Returns:
        plotly.graph_objects.Figure: The visualization figure
    """
    # Filter data based on selected year
    if selected_year != 'all':
        filtered_df = data[data['Year'] == selected_year]
    else:
        filtered_df = data
    
    # Create the visualization
    fig = px.line(
        filtered_df,
        x='Date',  # Replace with your actual date column
        y='Monthly_Active_Users',  # Replace with your actual user count column
        title=f'Spotify Monthly Active Users ({selected_year if selected_year != "all" else "All Years"})',
        markers=True,
        color_discrete_sequence=['#1DB954']  # Spotify green
    )
    
    # Customize the layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Monthly Active Users (millions)",
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font_color='#191414'
    )
    
    return fig 