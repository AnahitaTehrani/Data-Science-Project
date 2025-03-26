import os
import pandas as pd
import datetime
import glob
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create output directory for the CSV files
output_dir = os.path.join(script_dir, "news_analysis")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Load the CSV files
csv_files = glob.glob(os.path.join("/Users/armandocriscuolo/c2025/data_science_project_2025/code/Data-Science-Project/Data-Question-2-B/spotify_news_2024_2025_finhub", "*.csv"))

# Load and combine all CSV files
dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Combine all dataframes into one
df = pd.concat(dfs, ignore_index=True)

#print(df.head())

# Save the combined dataframe to a CSV file
df.to_csv(os.path.join(output_dir, "spotify_news_2024_2025_finhub.csv"), index=False)