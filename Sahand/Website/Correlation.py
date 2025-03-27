# This code was made by me with the help from Chat gpt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# 📥 Load CSV files (try both comma and semicolon as delimiters)
def load_csv_correctly(file_path):
    """Loads a CSV file and checks if the delimiter is correct."""
    try:
        df_comma = pd.read_csv(file_path, sep=",", encoding="utf-8")
        if len(df_comma.columns) > 1:
            return df_comma  # File correctly separated by comma
    except Exception:
        pass  # If it fails, try semicolon

    df_semicolon = pd.read_csv(file_path, sep=";", encoding="utf-8")
    if len(df_semicolon.columns) > 1:
        return df_semicolon  # File correctly separated by semicolon

    # If nothing works, just load it as-is
    return pd.read_csv(file_path, encoding="utf-8")


# 📌 Load properly named files
tracks_df = load_csv_correctly("playlist_tracks_cleaned.csv")
playlists_df = load_csv_correctly("spotify_top_playlists.csv")

# 🔍 Check if columns are correctly recognized
print("📌 Columns in tracks_df:", tracks_df.columns)
print("📌 Columns in playlists_df:", playlists_df.columns)

# 🔄 If `followers` is missing, try to use an alternative column
if "followers" not in playlists_df.columns:
    possible_names = ["Follower", "listener_count"]
    for name in possible_names:
        if name in playlists_df.columns:
            playlists_df.rename(columns={name: "followers"}, inplace=True)
            break
    else:
        # If no suitable column exists, set followers to 0
        playlists_df["followers"] = 0

# 🔄 If `playlist_id` is missing, try to extract it from the first row
if "playlist_id" not in playlists_df.columns:
    playlists_df.columns = playlists_df.columns[0].split(",")  # Fix misread columns

# 🎵 Group tracks to count how often a track appears in playlists
track_counts = tracks_df.groupby("track_id")["playlist_id"].count().reset_index()
track_counts.columns = ["track_id", "playlist_count"]

# 🎶 Add track names and artist names
track_counts = track_counts.merge(
    tracks_df[["track_id", "track_name", "artist_name"]].drop_duplicates(),
    on="track_id"
)

# 📊 Merge playlist follower data with track data
merged_df = tracks_df.merge(playlists_df[["playlist_id", "followers"]], on="playlist_id", how="left")

# 🏆 Calculate total followers per track
track_followers = merged_df.groupby("track_id")["followers"].sum().reset_index()
track_followers.columns = ["track_id", "total_followers"]

# 🔗 Combine data: number of playlists + total followers
df_final = track_counts.merge(track_followers, on="track_id")

# 📌 Show first few rows of the final table
print(df_final.head())

# 📈 Calculate Pearson correlation coefficient
corr, _ = pearsonr(df_final["playlist_count"], df_final["total_followers"])
print(f"📊 Pearson correlation coefficient: {corr:.2f}")

# 📊 Visualization: Is there a correlation?
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df_final["playlist_count"], y=df_final["total_followers"])
plt.xlabel("Number of Playlists")
plt.ylabel("Total Followers of Playlists")
plt.title("Correlation Between Playlist Frequency and Total Followers")
plt.show()
