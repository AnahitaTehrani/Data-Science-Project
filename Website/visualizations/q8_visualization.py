# q8_visualization.py
# Visualization for Research Question 8: Is there a tendency for popular tracks to appear more frequently in playlists?

import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr

def create_track_popularity_visualization(tracks_df, playlists_df):
    """Create the visualization for Question 8.
    
    Parameters:
    tracks_df (DataFrame): DataFrame containing track data
    playlists_df (DataFrame): DataFrame containing playlist data
    
    Returns:
    fig: A plotly figure object
    """
    try:
        # Ensure required columns exist
        required_tracks_cols = ["track_id", "playlist_id", "track_name", "artist_name"]
        required_playlists_cols = ["playlist_id", "followers"]
        
        # Normalize column names (handle capitalization issues)
        tracks_df.columns = [col.lower() for col in tracks_df.columns]
        playlists_df.columns = [col.lower() for col in playlists_df.columns]
        
        # Check for and rename columns if necessary
        if "followers" not in playlists_df.columns:
            possible_names = ["follower", "listener_count", "listeners"]
            for name in possible_names:
                if name in playlists_df.columns:
                    playlists_df = playlists_df.rename(columns={name: "followers"})
                    break
            else:
                # If no suitable column exists, create a placeholder
                playlists_df["followers"] = 100  # Default value
        
        # Validate the required columns
        for df, cols, name in [(tracks_df, required_tracks_cols, "tracks"), 
                               (playlists_df, required_playlists_cols, "playlists")]:
            missing = [col for col in cols if col not in df.columns]
            if missing:
                return {
                    'data': [],
                    'layout': {
                        'title': f"Missing columns in {name} data: {', '.join(missing)}",
                        'height': 500
                    }
                }
        
        # Process the data
        # Group tracks to count how often a track appears in playlists
        track_counts = tracks_df.groupby("track_id")["playlist_id"].count().reset_index()
        track_counts.columns = ["track_id", "playlist_count"]

        # Add track names and artist names
        track_counts = track_counts.merge(
            tracks_df[["track_id", "track_name", "artist_name"]].drop_duplicates(),
            on="track_id"
        )

        # Merge playlist follower data with track data
        merged_df = tracks_df.merge(playlists_df[["playlist_id", "followers"]], on="playlist_id", how="left")

        # Calculate total followers per track
        track_followers = merged_df.groupby("track_id")["followers"].sum().reset_index()
        track_followers.columns = ["track_id", "total_followers"]

        # Combine data: number of playlists + total followers
        df_final = track_counts.merge(track_followers, on="track_id")
        
        # Calculate correlation
        correlation, _ = pearsonr(df_final["playlist_count"], df_final["total_followers"])
        
        # Create the visualization
        fig = px.scatter(
            df_final, 
            x="playlist_count", 
            y="total_followers",
            hover_data=["track_name", "artist_name"],
            title=f"Correlation Between Playlist Frequency and Total Followers (r = {correlation:.2f})",
            labels={
                "playlist_count": "Number of Playlists",
                "total_followers": "Total Followers of Playlists",
                "track_name": "Track",
                "artist_name": "Artist"
            },
            template="plotly_white"
        )
        
        # Add trendline
        trendline = px.scatter(
            df_final, 
            x="playlist_count", 
            y="total_followers", 
            trendline="ols"
        ).data[1]
        
        fig.add_trace(trendline)
        
        # Customize layout
        fig.update_traces(marker=dict(size=8, color="#1DB954"))  # Spotify green
        fig.update_layout(
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            plot_bgcolor="white",
            height=550,
        )
        
        return fig
        
    except Exception as e:
        # Return empty figure with error message
        return {
            'data': [],
            'layout': {
                'title': f"Error processing data: {str(e)}",
                'height': 500
            }
        }