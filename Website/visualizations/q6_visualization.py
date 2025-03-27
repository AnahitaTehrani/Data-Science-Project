# Implement visualizations for Question 6: Billboard vs Spotify Chart Comparison

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
import dash_bootstrap_components as dbc

def load_data():
    # Load Billboard and Spotify chart data
    billboard_df = pd.read_csv('data/Billboard-global-weekly-2024-08-10.csv', sep=';')
    spotify_df = pd.read_csv('data/regional-global-weekly-2024-08-08.csv')
    
    # Clean and prepare data
    billboard_df = billboard_df[["rank", "artist_names", "track_name", "previous_rank", "peak_rank", "weeks_on_chart"]]
    spotify_df = spotify_df[["rank", "artist_names", "track_name", "previous_rank", "peak_rank", "weeks_on_chart"]]
    
    # Create a merged dataset for shared songs
    spotify_df.rename(columns={
        'rank': 'rank_spotify', 
        'weeks_on_chart': 'weeks_spotify', 
        "previous_rank": "previous_spotify", 
        "peak_rank":"peak_spotify"
    }, inplace=True)
    
    billboard_df.rename(columns={
        'rank': 'rank_billboard', 
        'weeks_on_chart': 'weeks_billboard',
        "previous_rank": "previous_billboard",
        "peak_rank":"peak_billboard"
    }, inplace=True)
    
    shared_songs = pd.merge(billboard_df, spotify_df, on=["artist_names", "track_name"], suffixes=("_billboard", "_spotify"))
    shared_songs["Rank_Differ"] = abs(shared_songs["rank_billboard"] - shared_songs["rank_spotify"])
    
    return billboard_df, spotify_df, shared_songs

def create_chart_overlap_fig():
    billboard_df, spotify_df, shared_songs = load_data()
    
    # Calculate metrics
    billboard_songs = set(zip(billboard_df["artist_names"], billboard_df["track_name"]))
    spotify_songs = set(zip(spotify_df["artist_names"], spotify_df["track_name"]))
    shared_songs_set = billboard_songs.intersection(spotify_songs)
    
    billboard_only = len(billboard_songs) - len(shared_songs_set)
    spotify_only = len(spotify_songs) - len(shared_songs_set)
    shared = len(shared_songs_set)
    
    # Create pie chart for song overlap
    labels = ["shared songs", "only in Billboard", "only in spotify"]
    sizes = [shared, billboard_only, spotify_only]
    colors = ["red", "blue", "yellow"]
    
    fig = px.pie(
        values=sizes, 
        names=labels, 
        title="Song Overlap Between Billboard and Spotify Charts",
        color_discrete_sequence=colors
    )
    
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(height=500)
    
    return fig

def create_artist_overlap_fig():
    billboard_df, spotify_df, shared_songs = load_data()
    
    # Calculate artist overlap
    billboard_artists = set(billboard_df["artist_names"])
    spotify_artists = set(spotify_df["artist_names"])
    shared_artists = billboard_artists.intersection(spotify_artists)
    
    billboard_only = len(billboard_artists) - len(shared_artists)
    spotify_only = len(spotify_artists) - len(shared_artists)
    shared = len(shared_artists)
    
    # Create pie chart for artist overlap
    labels = ["shared artists", "only in Billboard", "only in spotify"]
    sizes = [shared, billboard_only, spotify_only]
    colors = ["purple", "green", "orange"]
    
    fig = px.pie(
        values=sizes, 
        names=labels, 
        title="Artist Overlap Between Billboard and Spotify Charts",
        color_discrete_sequence=colors
    )
    
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(height=500)
    
    return fig

def create_correlation_fig():
    billboard_df, spotify_df, shared_songs = load_data()
    
    # Calculate correlations
    correlations = {
        "Ranking": shared_songs["rank_spotify"].corr(shared_songs["rank_billboard"]),
        "Weeks": shared_songs["weeks_spotify"].corr(shared_songs["weeks_billboard"]),
        "Previous Rank": shared_songs["previous_spotify"].corr(shared_songs["previous_billboard"]),
        "Peak Rank": shared_songs["peak_spotify"].corr(shared_songs["peak_billboard"])
    }
    
    # Create scatter plot with dropdown for different correlations
    fig = px.scatter(
        shared_songs,
        x="rank_spotify",
        y="rank_billboard",
        title=f"Correlation between Spotify and Billboard Rankings: {correlations['Ranking']:.2f}",
        labels={"rank_spotify": "Spotify Rank", "rank_billboard": "Billboard Rank"},
        height=600
    )
    
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": f"Ranking (r= {correlations['Ranking']:.2f})",
                        "method": "update",
                        "args": [{"x": shared_songs["rank_spotify"], "y": shared_songs["rank_billboard"]},
                                {"title": f"Ranking Correlation: {correlations['Ranking']:.2f}"}]
                    },
                    {
                        "label": f"Weeks on Chart (r = {correlations['Weeks']:.2f})",
                        "method": "update",
                        "args": [{"x": shared_songs["weeks_spotify"], "y": shared_songs["weeks_billboard"]},
                                {"title": f"Weeks on Chart Correlation: {correlations['Weeks']:.2f}"}]
                    },
                    {
                        "label": f"Previous Rank (r = {correlations['Previous Rank']:.2f})",
                        "method": "update",
                        "args": [{"x": shared_songs["previous_spotify"], "y": shared_songs["previous_billboard"]},
                                {"title": f"Previous Rank Correlation: {correlations['Previous Rank']:.2f}"}]
                    },
                    {
                        "label": f"Peak Rank (r = {correlations['Peak Rank']:.2f})",
                        "method": "update",
                        "args": [{"x": shared_songs["peak_spotify"], "y": shared_songs["peak_billboard"]},
                                {"title": f"Peak Rank Correlation: {correlations['Peak Rank']:.2f}"}]
                    }
                ],
                'direction': 'down',
                'showactive': True
            }
        ]
    )
    
    return fig

def render_question6():
    billboard_df, spotify_df, shared_songs = load_data()
    
    # Calculate metrics for text display
    similar_percent = len(shared_songs) / len(billboard_df) * 100
    avg_rank_diff = shared_songs["Rank_Differ"].mean()
    
    return html.Div([
        html.H2("How do the popular music charts in general compare to the popular Spotify music charts?", className="question-title"),
        
        # Text content from Answer.pdf
        html.Div([
            html.H3("Comparison of Billboard and Spotify Charts"),
            html.P("When comparing popular music charts like Billboard to Spotify's charts, there are both similarities and differences. The pie chart shows that almost half (48.6%) of the artists are on both charts, meaning they are popular on both platforms. However, 27.1% of the artists are only on Billboard, and 24.3% are only on Spotify. This shows that each platform ranks music differently."),
            html.P("Comparing the overlap of songs between Spotify and Billboard, there are more differences than similarities. The songs that appear on only one chart make up the same percentage of 39.2% each, while the songs that appear on both charts account for 21.6%. This shows that although some songs are popular on both platforms, many are unique to either Spotify or Billboard."),
            html.P(f"Furthermore, 35.5% of the songs that appear on both charts hold the same rank on each platform, with an average placement difference of {avg_rank_diff:.2f}. This shows that although some songs are popular on both platforms, many are unique to either Spotify or Billboard.")
        ], className="text-explanation"),
        
        # Song Overlap Chart
        html.Div([
            html.H3("Song Overlap Between Charts", className="chart-title"),
            dcc.Graph(figure=create_chart_overlap_fig())
        ], className="chart-container"),
        
        # Artist Overlap Chart
        html.Div([
            html.H3("Artist Overlap Between Charts", className="chart-title"),
            dcc.Graph(figure=create_artist_overlap_fig())
        ], className="chart-container"),
        
        # Correlation Analysis
        html.Div([
            html.H3("Chart Correlation Analysis", className="chart-title"),
            html.P("The Billboard and Spotify charts show important details about each song, such as its current rank, how many weeks it has been on the chart, its highest rank, and the previous rank. These four aspects help track how a song performs over time."),
            html.P("By comparing these aspects, we can see how similar the two charts are. Use the dropdown in the chart to explore different correlations:"),
            dcc.Graph(figure=create_correlation_fig())
        ], className="chart-container"),
    ])