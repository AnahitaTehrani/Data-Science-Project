# Implement visualizations for Question 3: Regional Spotify Activity

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
import dash_bootstrap_components as dbc

def load_data():
    # Load the data for regional Spotify activity
    df = pd.read_csv('data/Spotify_maus_year.csv', sep=';')
    df['Year'] = df['Year'].astype(int)
    return df

def create_regional_activity_fig(df):
    # Calculate average MAU by Year and Region
    df_avg = df.groupby(["Year", "Region"])["Mau (in %)"].mean().unstack()
    
    # Create bar chart
    fig = px.bar(
        df_avg, 
        barmode='group',
        title='Spotify Monthly Active Users (MAU) by Region (2019-2024)',
        labels={'value': 'annual average of Mau in %', 'Year': 'Year'},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    
    fig.update_layout(
        legend_title="Region",
        xaxis=dict(
            tickmode='linear',
            dtick=1
        ),
        plot_bgcolor='white',
        height=600
    )
    
    return fig

def create_growth_rate_fig(df):
    # Calculate growth rate
    df_avg = df.groupby(["Year", "Region"])["Mau (in %)"].mean().unstack()
    growth_rate = df_avg.pct_change(axis="index") * 100
    growth_rate.loc[2018] = 0  # Starting point reference
    
    # Create line chart
    fig = px.line(
        growth_rate.reset_index().melt(id_vars="Year"),
        x="Year",
        y="value",
        color="Region",
        title='Spotify User Growth Rate by Region (2018-2024)',
        labels={'value': 'growth Rate (in %)', 'Year': 'Year'},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    
    fig.update_layout(
        legend_title="Region",
        xaxis=dict(
            tickmode='linear',
            dtick=1
        ),
        plot_bgcolor='white',
        height=600
    )
    
    return fig

def create_map_visualization(df, year):
    # Filter data for the selected year
    year_data = df[df['Year'] == year]
    
    # Calculate average per region for the selected year
    region_avg = year_data.groupby('Region')['Mau (in %)'].mean().reset_index()
    
    # Map regions to actual geographic locations (approximate central points)
    continent_coords = {
        "Europe": {"lat": 54, "lon": 10},
        "Latin America": {"lat": -10, "lon": -60},
        "North America": {"lat": 50, "lon": -100},
        "RoW": {"lat": 0, "lon": 20}  # Rest of World
    }
    
    # Create expanded dataframe for visualization
    expanded_data = []
    for _, row in region_avg.iterrows():
        region = row['Region']
        coords = continent_coords.get(region, {"lat": 0, "lon": 0})
        expanded_data.append({
            "Region": region,
            "Mau (in %)": row['Mau (in %)'],
            "lat": coords["lat"],
            "lon": coords["lon"]
        })
    
    expanded_df = pd.DataFrame(expanded_data)
    
    # Create the map figure
    fig = px.scatter_geo(
        expanded_df,
        lat="lat",
        lon="lon",
        size="Mau (in %)",
        color="Region",
        text="Region",
        title=f"Spotify Monthly Active Users by Region ({year})",
        projection="natural earth",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    
    fig.update_traces(marker=dict(opacity=0.8))
    fig.update_layout(height=600)
    
    return fig

def render_question3():
    df = load_data()
    years = sorted(df['Year'].unique())
    
    return html.Div([
        html.H2("Which global regions show the highest Spotify streaming activity, and how has this changed over time?", className="question-title"),
        
        # Text content from Answer.pdf
        html.Div([
            html.P("Spotify is most popular in Europe and North America, as these regions have the highest number of active users. However, North America's user engagement has been slowly decreasing, while Europe remains strong."),
            html.P("The growth rate chart shows that Latin America and the Rest of the World (RoW) have had ups and downs, with RoW (which includes Asia, Africa, and other regions) seeing the biggest spikes in growth. This suggests that Spotify is growing fast in places like Asia and Africa. At the same time, North America's growth has been negative, meaning fewer new users or some leaving the platform."),
            html.P("Overall, Europe and North America have the most users, but Asia, Africa, and Latin America are becoming more important as Spotify expands.")
        ], className="text-explanation"),
        
        # Bar Chart
        html.Div([
            html.H3("Monthly Active Users by Region", className="chart-title"),
            dcc.Graph(figure=create_regional_activity_fig(df))
        ], className="chart-container"),
        
        # Growth Rate Chart
        html.Div([
            html.H3("User Growth Rate by Region", className="chart-title"),
            dcc.Graph(figure=create_growth_rate_fig(df))
        ], className="chart-container"),
        
        # Interactive Map
        html.Div([
            html.H3("Regional Activity Map", className="chart-title"),
            html.P("Select a year to see regional Spotify activity:"),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(year), 'value': year} for year in years],
                value=years[-1],  # Default to most recent year
                clearable=False,
                className="dropdown"
            ),
            dcc.Graph(id='map-visualization')
        ], className="chart-container"),
    ])

# Callback function to be defined in app.py
def update_map(selected_year):
    df = load_data()
    return create_map_visualization(df, selected_year)