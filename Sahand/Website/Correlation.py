# This code was made by me with the help from ChatGPT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Loading CSV files - trying different separators because some files are weird
def load_csv_correctly(file_path):
   """Tries to load a CSV file with comma or semicolon separator."""
   try:
       df_comma = pd.read_csv(file_path, sep=",", encoding="utf-8")
       if len(df_comma.columns) > 1:
           return df_comma  # Comma worked
   except Exception:
       pass  # if Failed, will try semicolon

   df_semicolon = pd.read_csv(file_path, sep=";", encoding="utf-8")
   if len(df_semicolon.columns) > 1:
       return df_semicolon  # Semicolon worked

   
   return pd.read_csv(file_path, encoding="utf-8")


# Loading my data files
tracks_df = load_csv_correctly("playlist_tracks_cleaned.csv")
playlists_df = load_csv_correctly("spotify_top_playlists.csv")

# Checking if my columns are right
print("Columns in tracks_df:", tracks_df.columns)
print("Columns in playlists_df:", playlists_df.columns)

# Fixing column names if they're not what I expect
if "followers" not in playlists_df.columns:
   possible_names = ["Follower", "listener_count"]
   for name in possible_names:
       if name in playlists_df.columns:
           playlists_df.rename(columns={name: "followers"}, inplace=True)
           break
   else:
       
       playlists_df["followers"] = 0

# More column fixing
if "playlist_id" not in playlists_df.columns:
   playlists_df.columns = playlists_df.columns[0].split(",")  # Trying to fix bad columns

# Counting how many playlists each track appears in
track_counts = tracks_df.groupby("track_id")["playlist_id"].count().reset_index()
track_counts.columns = ["track_id", "playlist_count"]

# Adding track names and artists
track_counts = track_counts.merge(
   tracks_df[["track_id", "track_name", "artist_name"]].drop_duplicates(),
   on="track_id"
)

# Combining playlist follower data with tracks
merged_df = tracks_df.merge(playlists_df[["playlist_id", "followers"]], on="playlist_id", how="left")

# Adding up all followers for each track
track_followers = merged_df.groupby("track_id")["followers"].sum().reset_index()
track_followers.columns = ["track_id", "total_followers"]

# Putting everything together in one table
df_final = track_counts.merge(track_followers, on="track_id")

# Looking at the first few rows to check
print(df_final.head())

# Calculating correlation to see if there's a relationship
corr, _ = pearsonr(df_final["playlist_count"], df_final["total_followers"])
print(f"Pearson correlation coefficient: {corr:.2f}")

# Making a chart to see the relationship
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df_final["playlist_count"], y=df_final["total_followers"])
plt.xlabel("Number of Playlists")
plt.ylabel("Total Followers of Playlists")
plt.title("Correlation Between Playlist Frequency and Total Followers")
plt.show()
