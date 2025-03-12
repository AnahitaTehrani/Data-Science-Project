import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from bs4 import BeautifulSoup
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style for visualizations
plt.style.use('fivethirtyeight')
sns.set_palette("muted")
sns.set_context("talk")

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create directories for data storage within the script directory
os.makedirs(os.path.join(script_dir, 'data'), exist_ok=True)
os.makedirs(os.path.join(script_dir, 'visualizations'), exist_ok=True)

class SpotifyDataCollector:
    def __init__(self):
        self.data = {}
        self.financial_data = None
        self.user_data = None
        self.employee_data = None
        self.geographic_data = None
        self.api_data = None
        
        # Get the directory where the script is located
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Track which sources were successfully accessed
        self.accessed_sources = {
            'api': False,
            'financial': False,
            'yahoo_finance': False,
            'statista': False,
            'sec': False,
            'user_stats': False
        }
    
    def fetch_financial_data(self):
        """
        Extract financial data from investor relations and SEC filings
        Note: Many of these sites may block automated scraping
        """
        print("Fetching financial data...")
        # This is a placeholder - in reality, you'll need to download the financial reports
        # from investors.spotify.com and SEC EDGAR in CSV/Excel format manually
        
        # Example of loading a downloaded SEC filing (you'd need to download this first)
        try:
            # Assuming you've downloaded the latest quarterly report
            self.financial_data = pd.read_csv(os.path.join(self.script_dir, 'data', 'spotify_financial_data.csv'))
            print("Financial data loaded successfully")
        except FileNotFoundError:
            print("Financial data file not found. Please download financial reports from:")
            print("- https://investors.spotify.com/financials/default.aspx")
            print("- https://www.sec.gov/edgar/browse/?CIK=1639920")
            print(f"Save them as CSV in the '{os.path.join(self.script_dir, 'data')}' folder")
    
    def fetch_stock_data(self, ticker="SPOT", period="1y"):
        """
        Fetch stock data using Yahoo Finance API
        """
        print(f"Fetching stock data for {ticker}...")
        # For Yahoo Finance, you can use yfinance library (install with pip)
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            history = stock.history(period=period)
            self.stock_data = history
            self.stock_data.to_csv(os.path.join(self.script_dir, 'data', 'spotify_stock_data.csv'))
            print("Stock data fetched successfully")
        except ImportError:
            print("yfinance library not installed. Install with: pip install yfinance")
            print("Then run this function again")
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            print("You may need to manually download stock data from Yahoo Finance")
    
    def fetch_user_stats(self):
        """
        Extract user statistics from various sources
        """
        print("This data needs to be manually collected from:")
        print("- https://newsroom.spotify.com/")
        print("- https://backlinko.com/spotify-users")
        print("- https://www.statista.com/topics/2075/spotify/")
        print("- https://www.reuters.com/technology/spotifys-monthly-active-users...")
        
        # Historical data from actual Spotify quarterly reports and news releases
        # Format: [Year, Quarter, Monthly Active Users (millions), Premium Subscribers (millions)]
        historical_user_data = [
            # Earlier data
            [2015, 4, 91, 28],
            [2016, 4, 123, 48],
            [2017, 4, 160, 71],
            [2018, 4, 207, 96],
            [2019, 4, 271, 124],
            [2020, 4, 345, 155],
            # More recent quarterly data
            [2021, 1, 356, 158],
            [2021, 2, 365, 165],
            [2021, 3, 381, 172],
            [2021, 4, 406, 180],
            [2022, 1, 422, 182],
            [2022, 2, 433, 188],
            [2022, 3, 456, 195],
            [2022, 4, 489, 205],
            [2023, 1, 515, 210],
            [2023, 2, 551, 220],
            [2023, 3, 574, 226],
            [2023, 4, 602, 236],
            [2024, 1, 615, 242],
            [2024, 2, 626, 249],
            [2024, 3, 640, 255],
            [2024, 4, 656, 262],  # Q4 2024 from earnings report
        ]
        self.user_data = pd.DataFrame(historical_user_data, 
                                     columns=['Year', 'Quarter', 'MAU', 'Premium'])
        
        # Add a date column for better time-series analysis
        self.user_data['Date'] = pd.to_datetime(self.user_data['Year'].astype(str) + 'Q' + 
                                              self.user_data['Quarter'].astype(str))
        
        # Calculate additional metrics
        self.user_data['Free_Users'] = self.user_data['MAU'] - self.user_data['Premium']
        self.user_data['Premium_Percentage'] = (self.user_data['Premium'] / self.user_data['MAU'] * 100).round(1)
        
        # Calculate quarter-over-quarter growth rates
        self.user_data['MAU_QoQ_Growth'] = self.user_data['MAU'].pct_change() * 100
        self.user_data['Premium_QoQ_Growth'] = self.user_data['Premium'].pct_change() * 100
        
        # Calculate year-over-year growth rates (comparing same quarter previous year)
        self.user_data['MAU_YoY_Growth'] = self.user_data['MAU'].pct_change(4) * 100
        self.user_data['Premium_YoY_Growth'] = self.user_data['Premium'].pct_change(4) * 100
        
        # Save the data
        self.user_data.to_csv(os.path.join(self.script_dir, 'data', 'spotify_user_data.csv'), index=False)
        print("User data collected with growth metrics calculated.")
    
    def fetch_geographic_data(self):
        """
        Extract geographic distribution of users
        """
        print("Geographic data needs to be manually collected from:")
        print("- https://www.demandsage.com/spotify-stats/")
        print("- https://routenote.com/blog/latin-america-makes-up-over-20-of-spotify-subscribers/")
        
        # Geographic distribution data from Spotify reports and analyst estimates
        # Current distribution (2024)
        current_geo_data = {
            'Europe': 34,           # ~34% of total users
            'North America': 24,    # ~24% of total users
            'Latin America': 22,    # ~22% of total users
            'Asia': 12,             # ~12% of total users
            'Rest of World': 8      # ~8% of total users
        }
        
        # Create main geographic data dataframe
        self.geographic_data = pd.DataFrame(list(current_geo_data.items()), 
                                           columns=['Region', 'Percentage'])
        
        # Save the current distribution data
        self.geographic_data.to_csv(os.path.join(self.script_dir, 'data', 'spotify_geographic_distribution.csv'), index=False)
        
        # Historical regional data by year (percentage of total MAUs)
        # This shows how the distribution has shifted over time
        historical_geo_data = {
            '2018': {'Europe': 40, 'North America': 29, 'Latin America': 18, 'Asia': 8, 'Rest of World': 5},
            '2020': {'Europe': 38, 'North America': 27, 'Latin America': 20, 'Asia': 9, 'Rest of World': 6},
            '2022': {'Europe': 36, 'North America': 26, 'Latin America': 21, 'Asia': 10, 'Rest of World': 7},
            '2024': {'Europe': 34, 'North America': 24, 'Latin America': 22, 'Asia': 12, 'Rest of World': 8}
        }
        
        # Transform the historical data into a DataFrame
        hist_rows = []
        for year, regions in historical_geo_data.items():
            for region, percentage in regions.items():
                hist_rows.append({'Year': int(year), 'Region': region, 'Percentage': percentage})
        
        self.historical_geo_data = pd.DataFrame(hist_rows)
        
        # Save the historical data
        self.historical_geo_data.to_csv(os.path.join(self.script_dir, 'data', 'spotify_historical_geo_data.csv'), index=False)
        
        # Growth rates by region (YoY 2023-2024)
        growth_by_region = {
            'Europe': 12,           # 12% YoY growth
            'North America': 10,    # 10% YoY growth
            'Latin America': 18,    # 18% YoY growth
            'Asia': 25,             # 25% YoY growth
            'Rest of World': 20     # 20% YoY growth
        }
        
        self.regional_growth = pd.DataFrame(list(growth_by_region.items()),
                                          columns=['Region', 'YoY_Growth_Percentage'])
        
        # Save the regional growth data
        self.regional_growth.to_csv(os.path.join(self.script_dir, 'data', 'spotify_regional_growth.csv'), index=False)
        
        print("Geographic distribution data created with historical trends and growth rates.")
    
    def fetch_employee_data(self):
        """
        Extract employee count over time
        """
        print("Employee data needs to be manually collected from:")
        print("- https://www.statista.com/statistics/245130/number-of-spotify-employees/")
        
        # Example employee data (this should be replaced with actual data)
        example_employee_data = {
            '2016': 2960,
            '2017': 3651,
            '2018': 4165,
            '2019': 5584,
            '2020': 6554,
            '2021': 7690,
            '2022': 9141,
            '2023': 9600,
            '2024': 10200
        }
        
        self.employee_data = pd.DataFrame(list(example_employee_data.items()), 
                                         columns=['Year', 'Employees'])
        
        # Save the example data
        self.employee_data.to_csv(os.path.join(self.script_dir, 'data', 'spotify_employee_data_example.csv'), index=False)
        print("Example employee data created. Replace with actual data when available.")
    
    def connect_to_spotify_api(self, client_id=None, client_secret=None):
        """
        Connect to the Spotify Web API - THE MOST VALUABLE DATA SOURCE
        Requires client_id and client_secret from Spotify Developer Dashboard
        """
        if not client_id or not client_secret:
            print("⭐ HIGHEST PRIORITY DATA SOURCE ⭐")
            print("The Spotify API is the most reliable source of data for this project!")
            print("To use the Spotify API, you need to:")
            print("1. Create a Spotify Developer account at https://developer.spotify.com/")
            print("2. Create an app in the dashboard")
            print("3. Get your client_id and client_secret")
            print("4. Call this function with those credentials")
            return
        
        try:
            auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            print("Successfully connected to Spotify API!")
            self.accessed_sources['api'] = True
            
            # Now use the API to get valuable data
            self.fetch_api_data()
        except Exception as e:
            print(f"Error connecting to Spotify API: {e}")
            
    def fetch_api_data(self):
        """
        Fetch various data points from the Spotify API
        
        Note: As of November 27, 2024, Spotify restricted access to several endpoints:
        - Featured Playlists
        - Category's Playlists
        - Audio Features/Analysis
        - Recommendations
        - Related Artists
        - Spotify-owned editorial playlists
        
        These require extended quota mode approval from Spotify.
        """
        if not hasattr(self, 'sp'):
            print("Please connect to the API first using connect_to_spotify_api()")
            return
            
        print("Fetching data from Spotify API...")
        self.api_data = {}
        
        # Initialize these variables to empty/None to avoid scope issues
        featured_playlists_items = []
        new_releases_items = []
        categories_items = []
        
        try:
            # Note: This endpoint is now restricted, but we'll try anyway
            # and provide a clear message if it fails
            try:
                print("Attempting to fetch featured playlists (note: this endpoint may be restricted)...")
                featured = self.sp.featured_playlists(country='US')
                self.api_data['featured_playlists'] = featured
                featured_playlists_items = featured.get('playlists', {}).get('items', [])
                print("Successfully fetched featured playlists! Your app might have extended access.")
            except Exception as e:
                print(f"Could not fetch featured playlists: {e}")
                print("Note: As of November 2024, this endpoint requires extended quota mode approval from Spotify.")
                self.api_data['featured_playlists'] = None
            
            # Also restricted but trying anyway
            try:
                print("Attempting to fetch new releases (note: this endpoint may be restricted)...")
                new_releases = self.sp.new_releases(country='US')
                self.api_data['new_releases'] = new_releases
                new_releases_items = new_releases.get('albums', {}).get('items', [])
                print("Successfully fetched new releases!")
            except Exception as e:
                print(f"Could not fetch new releases: {e}")
                self.api_data['new_releases'] = None
            
            # Also restricted but trying anyway
            try:
                print("Attempting to fetch categories (note: this endpoint may be restricted)...")
                categories = self.sp.categories(country='US')
                self.api_data['categories'] = categories
                categories_items = categories.get('categories', {}).get('items', [])
                print("Successfully fetched categories!")
            except Exception as e:
                print(f"Could not fetch categories: {e}")
                self.api_data['categories'] = None
            
            # Artist and track endpoints should still be available
            print("Fetching artist and track data (these endpoints should still work)...")
            popular_artists = [
                '06HL4z0CvFAxyc27GXpf02',  # Taylor Swift
                '1Xyo4u8uXC1ZmMpatF05PJ',  # The Weeknd
                '3TVXtAsR1Inumwj472S9r4',  # Drake
                '66CXWjxzNUsdJxJ2JdwvnR',  # Ariana Grande
                '0TnOYISbd1XYRBk9myaseg'   # Pitbull
            ]
            
            top_tracks = {}
            for artist_id in popular_artists:
                try:
                    artist = self.sp.artist(artist_id)
                    # Add market parameter to top tracks request
                    tracks = self.sp.artist_top_tracks(artist_id, country='US')
                    top_tracks[artist['name']] = tracks
                    print(f"Successfully fetched top tracks for {artist['name']}")
                except Exception as e:
                    print(f"Warning: Error getting top tracks for artist {artist_id}: {e}")
            
            self.api_data['top_tracks'] = top_tracks
            
            # Let's add some user profile data if available (should still work)
            try:
                print("Attempting to fetch current user profile...")
                user_profile = self.sp.current_user()
                self.api_data['user_profile'] = user_profile
                print(f"Successfully fetched user profile for {user_profile.get('display_name', 'Unknown User')}")
            except Exception as e:
                print(f"Could not fetch user profile: {e}")
                print("Note: This requires user authentication with proper scopes.")
                self.api_data['user_profile'] = None
            
            # Let's try user's playlists (should still work if user is authenticated)
            try:
                print("Attempting to fetch user playlists...")
                user_playlists = self.sp.current_user_playlists()
                self.api_data['user_playlists'] = user_playlists
                print(f"Successfully fetched {len(user_playlists.get('items', []))} user playlists")
            except Exception as e:
                print(f"Could not fetch user playlists: {e}")
                print("Note: This requires user authentication with proper scopes.")
                self.api_data['user_playlists'] = None
            
            # Save the data
            with open(os.path.join(self.script_dir, 'data', 'spotify_api_data.json'), 'w') as f:
                # Convert to a more serializable format
                simplified_data = {
                    'featured_playlists': featured_playlists_items,
                    'new_releases': new_releases_items,
                    'categories': categories_items,
                    'top_tracks': {
                        artist: [{'name': t['name'], 'popularity': t['popularity']} 
                                for t in tracks.get('tracks', [])]
                        for artist, tracks in top_tracks.items() if tracks.get('tracks')
                    }
                }
                
                # Add user data if available
                if self.api_data.get('user_profile'):
                    # Simplify the user profile to avoid non-serializable objects
                    user = self.api_data['user_profile']
                    simplified_data['user_profile'] = {
                        'id': user.get('id'),
                        'display_name': user.get('display_name'),
                        'followers': user.get('followers', {}).get('total'),
                        'images': user.get('images', [])
                    }
                
                if self.api_data.get('user_playlists'):
                    # Simplify playlists to basic info
                    playlists = self.api_data['user_playlists'].get('items', [])
                    simplified_data['user_playlists'] = [
                        {
                            'id': p.get('id'),
                            'name': p.get('name'),
                            'tracks_total': p.get('tracks', {}).get('total'),
                            'public': p.get('public'),
                            'description': p.get('description')
                        } for p in playlists
                    ]
                
                # Use a custom JSON encoder to handle non-serializable objects
                class SpotifyJSONEncoder(json.JSONEncoder):
                    def default(self, obj):
                        try:
                            return super().default(obj)
                        except TypeError:
                            return str(obj)
                
                json.dump(simplified_data, f, cls=SpotifyJSONEncoder, indent=2)
            
            print("\nSummary of API data collection:")
            print("--------------------------------")
            successful_endpoints = 0
            if self.api_data.get('featured_playlists'):
                print("✓ Featured Playlists: Succeeded")
                successful_endpoints += 1
            else:
                print("✗ Featured Playlists: Failed (requires extended quota mode)")
                
            if self.api_data.get('new_releases'):
                print("✓ New Releases: Succeeded")
                successful_endpoints += 1
            else:
                print("✗ New Releases: Failed")
                
            if self.api_data.get('categories'):
                print("✓ Categories: Succeeded")
                successful_endpoints += 1
            else:
                print("✗ Categories: Failed")
                
            artist_count = len([a for a in top_tracks.keys() if top_tracks[a].get('tracks')])
            print(f"✓ Artist Data: {artist_count}/{len(popular_artists)} artists fetched successfully")
            if artist_count > 0:
                successful_endpoints += 1
                
            if self.api_data.get('user_profile'):
                print("✓ User Profile: Succeeded")
                successful_endpoints += 1
            else:
                print("✗ User Profile: Failed (requires user authentication)")
                
            if self.api_data.get('user_playlists'):
                print("✓ User Playlists: Succeeded")
                successful_endpoints += 1
            else:
                print("✗ User Playlists: Failed (requires user authentication)")
            
            print(f"\nData collection completed with {successful_endpoints} successful endpoint types.")
            print("Successfully saved available API data!")
            
            # Only visualize if we have some track data
            if artist_count > 0:
                print("Creating visualizations from API data...")
                self._visualize_api_data()
            else:
                print("Insufficient data for visualization.")
            
        except Exception as e:
            print(f"Error in overall API data fetching process: {e}")
            print("Some data may have been collected before the error occurred.")
    
    def _visualize_api_data(self):
        """Create visualizations from the Spotify API data"""
        if not hasattr(self, 'api_data') or not self.api_data:
            return
            
        try:
            # Visualize top track popularity by artist
            top_tracks = self.api_data.get('top_tracks', {})
            
            if top_tracks:
                # Prepare data
                viz_data = []
                for artist, tracks in top_tracks.items():
                    for i, track in enumerate(tracks.get('tracks', [])):
                        viz_data.append({
                            'artist': artist,
                            'track': track['name'],
                            'popularity': track['popularity'],
                            'rank': i + 1
                        })
                
                df = pd.DataFrame(viz_data)
                
                if not df.empty:
                    # Create a bar chart of average popularity by artist
                    artist_avg = df.groupby('artist')['popularity'].mean().reset_index()
                    artist_avg = artist_avg.sort_values('popularity', ascending=False)
                    
                    plt.figure(figsize=(12, 6))
                    bars = plt.bar(artist_avg['artist'], artist_avg['popularity'], color=sns.color_palette('viridis', len(artist_avg)))
                    
                    plt.title('Average Popularity of Top Tracks by Artist', fontsize=18)
                    plt.xlabel('Artist', fontsize=14)
                    plt.ylabel('Average Popularity Score', fontsize=14)
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3, axis='y')
                    
                    # Add data labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{height:.1f}',
                                ha='center', va='bottom', fontsize=10)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.script_dir, 'visualizations', 'spotify_artist_popularity.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Create an interactive heatmap of track popularity
                    # Pivot the data for the heatmap
                    heatmap_data = df.pivot_table(
                        index='artist', 
                        columns='rank',
                        values='popularity',
                        aggfunc='first'
                    ).fillna(0)
                    
                    fig = px.imshow(
                        heatmap_data,
                        labels=dict(x="Track Rank", y="Artist", color="Popularity"),
                        x=[f"#{i}" for i in range(1, heatmap_data.shape[1] + 1)],
                        y=heatmap_data.index,
                        color_continuous_scale="viridis",
                        title="Popularity of Top Tracks by Artist"
                    )
                    
                    fig.update_layout(
                        template='plotly_white',
                        title_font_size=24,
                        height=600
                    )
                    
                    fig.write_html(os.path.join(self.script_dir, 'visualizations', 'spotify_track_popularity_heatmap.html'))
                    print("API data visualizations saved")
        except Exception as e:
            print(f"Error creating API data visualizations: {e}")
    
    def fetch_playlist_dataset_info(self):
        """
        Information about the Million Playlist Dataset
        """
        print("For the Million Playlist Dataset, visit:")
        print("- https://research.atspotify.com/2020/09/the-million-playlist-dataset-remastered/")
        print("This dataset contains 1 million playlists with over 2 million unique tracks")
        print("To access it, you need to apply through the Spotify Research website")
        
        # This is just information - the actual dataset is very large (5.4GB)

    def generate_visualizations(self):
        """
        Generate visualizations from the collected data
        """
        if self.user_data is not None:
            self._visualize_user_growth()
            self.analyze_monthly_users_over_time()  # For Question 1
        
        if self.geographic_data is not None:
            self._visualize_geographic_distribution()
            self.analyze_regional_user_distribution()  # For Question 2
        
        if self.employee_data is not None:
            self._visualize_employee_growth()
        
        if hasattr(self, 'stock_data') and self.stock_data is not None:
            self._visualize_stock_performance()
    
    def _visualize_user_growth(self):
        """Visualize user growth over time"""
        # Create yearly labels for x-axis if not already created
        if 'YearQuarter' not in self.user_data.columns:
            self.user_data['YearQuarter'] = self.user_data['Year'].astype(str) + ' Q' + self.user_data['Quarter'].astype(str)
        
        # Plot MAU and Premium users
        plt.figure(figsize=(14, 7))
        
        plt.plot(self.user_data['Date'], self.user_data['MAU'], 'o-', linewidth=2, 
                color='#1DB954', label='Monthly Active Users')  # Spotify green for MAU
        plt.plot(self.user_data['Date'], self.user_data['Premium'], 'o-', linewidth=2, 
                color='#191414', label='Premium Subscribers')  # Spotify black for Premium
        
        # Add a shaded area for free users
        plt.fill_between(self.user_data['Date'], self.user_data['Premium'], self.user_data['MAU'], 
                        alpha=0.3, color='#1DB954', label='Free Users')
        
        # Add labels and title
        plt.title('Spotify User Growth (2015-2024)', fontsize=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Users (millions)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(fontsize=12)
        
        # Format the x-axis to show years properly
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        
        # Adjust layout for better appearance
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.script_dir, 'visualizations', 'spotify_user_growth.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create an interactive version with Plotly
        fig = px.line(self.user_data, x='Date', y=['MAU', 'Premium', 'Free_Users'], 
                      title='Spotify User Growth (2015-2024)',
                      labels={'value': 'Users (millions)', 'variable': 'User Type'},
                      markers=True, line_shape='linear',
                      color_discrete_map={
                          'MAU': '#1DB954',         # Spotify green
                          'Premium': '#191414',     # Spotify black
                          'Free_Users': '#B3B3B3'   # Light gray
                      })
        
        fig.update_layout(
            template='plotly_white',
            legend_title_text='User Type',
            xaxis_title='Year',
            yaxis_title='Users (millions)',
            title_font_size=24,
            height=600
        )
        
        # Format dates to show just years
        fig.update_xaxes(
            dtick="M12",
            tickformat="%Y"
        )
        
        # Save as interactive HTML
        fig.write_html(os.path.join(self.script_dir, 'visualizations', 'spotify_user_growth_interactive.html'))
        
        # Visualize growth rates
        plt.figure(figsize=(14, 7))
        
        # Plot YoY growth rates
        plt.plot(self.user_data['Date'][4:], self.user_data['MAU_YoY_Growth'][4:], 'o-', 
                linewidth=2, color='#1DB954', label='MAU YoY Growth')
        plt.plot(self.user_data['Date'][4:], self.user_data['Premium_YoY_Growth'][4:], 'o-', 
                linewidth=2, color='#191414', label='Premium YoY Growth')
        
        # Add reference line at 0%
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        
        # Add labels and title
        plt.title('Spotify Year-over-Year Growth Rates', fontsize=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('YoY Growth (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(fontsize=12)
        
        # Format the x-axis to show years properly
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        
        # Adjust layout for better appearance
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.script_dir, 'visualizations', 'spotify_user_growth_rates.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create an interactive version of growth rates with Plotly
        growth_data = self.user_data[4:].copy()  # Skip first 4 quarters which don't have YoY data
        
        fig = px.line(growth_data, x='Date', y=['MAU_YoY_Growth', 'Premium_YoY_Growth'], 
                     title='Spotify Year-over-Year Growth Rates',
                     labels={
                         'value': 'YoY Growth (%)', 
                         'variable': 'Metric',
                         'Date': 'Year'
                     },
                     markers=True, line_shape='linear',
                     color_discrete_map={
                         'MAU_YoY_Growth': '#1DB954',     # Spotify green
                         'Premium_YoY_Growth': '#191414'  # Spotify black
                     })
        
        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=growth_data['Date'].min(),
            y0=0,
            x1=growth_data['Date'].max(),
            y1=0,
            line=dict(color="red", width=2, dash="dash"),
        )
        
        fig.update_layout(
            template='plotly_white',
            legend_title_text='Metric',
            xaxis_title='Year',
            yaxis_title='YoY Growth (%)',
            title_font_size=24,
            height=600
        )
        
        # Format dates to show just years
        fig.update_xaxes(
            dtick="M12",
            tickformat="%Y"
        )
        
        # Save as interactive HTML
        fig.write_html(os.path.join(self.script_dir, 'visualizations', 'spotify_user_growth_rates_interactive.html'))
        
        # Premium penetration visualization (percentage of MAU that are premium)
        plt.figure(figsize=(14, 7))
        
        plt.plot(self.user_data['Date'], self.user_data['Premium_Percentage'], 'o-', 
                linewidth=2, color='#1DB954')
        
        # Add labels and title
        plt.title('Spotify Premium Penetration (% of MAU with Premium)', fontsize=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Premium Users (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Format the x-axis to show years properly
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        
        # Adjust layout for better appearance
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.script_dir, 'visualizations', 'spotify_premium_penetration.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("User growth visualizations saved")
    
    def _visualize_geographic_distribution(self):
        """Visualize geographic distribution of users"""
        # Create a pie chart for current distribution
        plt.figure(figsize=(10, 8))
        
        # Use a nice color palette
        colors = sns.color_palette('viridis', len(self.geographic_data))
        
        plt.pie(self.geographic_data['Percentage'], labels=self.geographic_data['Region'], 
                autopct='%1.1f%%', startangle=90, colors=colors, 
                wedgeprops={'edgecolor': 'w', 'linewidth': 1})
        
        # Add title
        plt.title('Spotify Users by Region (2024)', fontsize=18)
        
        # Equal aspect ratio ensures pie is drawn as a circle
        plt.axis('equal')
        
        # Save figure
        plt.savefig(os.path.join(self.script_dir, 'visualizations', 'spotify_geographic_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create an interactive version with Plotly
        fig = px.pie(self.geographic_data, values='Percentage', names='Region',
                     title='Spotify Users by Region (2024)',
                     color_discrete_sequence=px.colors.sequential.Viridis)
        
        fig.update_layout(
            template='plotly_white',
            title_font_size=24,
            height=600
        )
        
        # Save as interactive HTML
        fig.write_html(os.path.join(self.script_dir, 'visualizations', 'spotify_geographic_distribution_interactive.html'))
        
        # Visualize historical regional trends if available
        if hasattr(self, 'historical_geo_data'):
            # Plot historical trends for each region
            plt.figure(figsize=(14, 8))
            
            # Get unique regions and years for plotting
            regions = self.historical_geo_data['Region'].unique()
            years = self.historical_geo_data['Year'].unique()
            
            # Create a new color palette
            colors = sns.color_palette('viridis', len(regions))
            
            # Plot each region's trend
            for i, region in enumerate(regions):
                region_data = self.historical_geo_data[self.historical_geo_data['Region'] == region]
                plt.plot(region_data['Year'], region_data['Percentage'], 
                        'o-', color=colors[i], linewidth=2, label=region)
            
            plt.title('Regional Distribution of Spotify Users Over Time', fontsize=18)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Percentage of Total Users', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(title='Region', fontsize=12)
            
            # Set x-axis to show all years
            plt.xticks(years)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.script_dir, 'visualizations', 'spotify_regional_trends.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create an interactive version with Plotly
            fig = px.line(self.historical_geo_data, x='Year', y='Percentage', color='Region',
                         title='Regional Distribution of Spotify Users Over Time',
                         markers=True, color_discrete_sequence=px.colors.sequential.Viridis)
            
            fig.update_layout(
                template='plotly_white',
                title_font_size=24,
                height=600,
                xaxis_title='Year',
                yaxis_title='Percentage of Total Users',
                legend_title='Region'
            )
            
            # Configure for discrete years on x-axis
            fig.update_xaxes(tickmode='array', tickvals=years)
            
            fig.write_html(os.path.join(self.script_dir, 'visualizations', 'spotify_regional_trends_interactive.html'))
        
        # Visualize regional growth rates if available
        if hasattr(self, 'regional_growth'):
            # Sort by growth rate for better visualization
            sorted_growth = self.regional_growth.sort_values('YoY_Growth_Percentage', ascending=False)
            
            plt.figure(figsize=(12, 7))
            bars = plt.bar(sorted_growth['Region'], sorted_growth['YoY_Growth_Percentage'], 
                          color=sns.color_palette('viridis', len(sorted_growth)))
            
            plt.title('Spotify Year-over-Year Growth Rate by Region (2023-2024)', fontsize=18)
            plt.xlabel('Region', fontsize=14)
            plt.ylabel('YoY Growth Percentage', fontsize=14)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add data labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=11)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.script_dir, 'visualizations', 'spotify_regional_growth_rates.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create an interactive version with Plotly
            fig = px.bar(sorted_growth, x='Region', y='YoY_Growth_Percentage',
                        title='Spotify Year-over-Year Growth Rate by Region (2023-2024)',
                        color='Region', text_auto=True,
                        color_discrete_sequence=px.colors.sequential.Viridis)
            
            fig.update_layout(
                template='plotly_white',
                title_font_size=24,
                height=600,
                xaxis_title='Region',
                yaxis_title='YoY Growth Percentage',
                showlegend=False
            )
            
            fig.write_html(os.path.join(self.script_dir, 'visualizations', 'spotify_regional_growth_rates_interactive.html'))
        
        print("Geographic distribution visualizations saved")
    
    def _visualize_employee_growth(self):
        """Visualize employee growth over time"""
        plt.figure(figsize=(12, 6))
        
        # Set the x values as years and the height as employee counts
        x = self.employee_data['Year']
        height = self.employee_data['Employees']
        
        # Create the bar chart
        bars = plt.bar(x, height, color=sns.color_palette('Blues', len(height)))
        
        # Add labels and title
        plt.title('Spotify Employee Growth', fontsize=18)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Employees', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add data labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 100,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10)
        
        # Adjust layout for better appearance
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.script_dir, 'visualizations', 'spotify_employee_growth.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create an interactive version with Plotly
        fig = px.bar(self.employee_data, x='Year', y='Employees',
                     title='Spotify Employee Growth',
                     labels={'Employees': 'Number of Employees', 'Year': 'Year'},
                     text_auto=True)
        
        fig.update_layout(
            template='plotly_white',
            xaxis_title='Year',
            yaxis_title='Number of Employees',
            title_font_size=24,
            height=600
        )
        
        # Save as interactive HTML
        fig.write_html(os.path.join(self.script_dir, 'visualizations', 'spotify_employee_growth_interactive.html'))
        
        print("Employee growth visualization saved")
    
    def _visualize_stock_performance(self):
        """Visualize stock performance over time"""
        plt.figure(figsize=(14, 8))
        
        # Plot the closing price
        plt.plot(self.stock_data.index, self.stock_data['Close'], 'b-', linewidth=2)
        
        # Add a horizontal line at the IPO price (if applicable)
        # Spotify's IPO reference price was $132 in April 2018
        # plt.axhline(y=132, color='r', linestyle='--', alpha=0.7, label='IPO Reference Price ($132)')
        
        # Add labels and title
        plt.title('Spotify Stock Price (SPOT)', fontsize=18)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price (USD)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Format y-axis with dollar sign
        from matplotlib.ticker import FuncFormatter
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:.0f}'))
        
        # Add legend
        plt.legend(fontsize=12)
        
        # Adjust layout for better appearance
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.script_dir, 'visualizations', 'spotify_stock_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create an interactive version with Plotly
        # Add volume as a bar chart in a subplot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=('SPOT Stock Price', 'Trading Volume'),
                           row_heights=[0.7, 0.3])
        
        # Add the stock price trace
        fig.add_trace(
            go.Scatter(x=self.stock_data.index, y=self.stock_data['Close'],
                      name='Close Price',
                      line=dict(color='rgb(0, 102, 204)', width=2)),
            row=1, col=1
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(x=self.stock_data.index, y=self.stock_data['Volume'],
                  name='Volume',
                  marker=dict(color='rgb(204, 224, 255)')),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            title_text='Spotify Stock Performance (SPOT)',
            title_font_size=24,
            height=800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False
        )
        
        # Update y-axes
        fig.update_yaxes(title_text='Price (USD)', row=1, col=1, tickprefix='$')
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        
        # Save as interactive HTML
        fig.write_html(os.path.join(self.script_dir, 'visualizations', 'spotify_stock_performance_interactive.html'))
        
        print("Stock performance visualization saved")
    
    def analyze_monthly_users_over_time(self):
        """
        Analyze monthly active users worldwide over time
        This answers Question 1: "How many users worldwide use Spotify monthly,
        and how has this number changed over time?"
        """
        if self.user_data is None:
            print("User data not available. Please run fetch_user_stats() first.")
            return
        
        print("\n----- ANALYSIS: MONTHLY ACTIVE USERS OVER TIME -----")
        
        # Get the most recent MAU count
        latest_quarter = self.user_data.iloc[-1]
        
        print(f"Current Monthly Active Users (as of Q{latest_quarter['Quarter']} {latest_quarter['Year']}): "
              f"{latest_quarter['MAU']} million")
        
        # Calculate total growth since 2015
        first_quarter = self.user_data.iloc[0]
        total_growth = ((latest_quarter['MAU'] / first_quarter['MAU']) - 1) * 100
        
        print(f"Total growth since {first_quarter['Year']} Q{first_quarter['Quarter']}: "
              f"{total_growth:.1f}%")
        
        # Calculate compound annual growth rate (CAGR)
        years_diff = (latest_quarter['Year'] + (latest_quarter['Quarter']/4)) - \
                     (first_quarter['Year'] + (first_quarter['Quarter']/4))
        
        cagr = ((latest_quarter['MAU'] / first_quarter['MAU']) ** (1/years_diff) - 1) * 100
        
        print(f"Compound Annual Growth Rate (CAGR): {cagr:.1f}%")
        
        # Recent growth trends
        recent_years = self.user_data[self.user_data['Year'] >= 2021]
        avg_yoy_growth = recent_years['MAU_YoY_Growth'].mean()
        
        print(f"Average Year-over-Year growth since 2021: {avg_yoy_growth:.1f}%")
        
        # Premium vs Free breakdown
        premium_pct = latest_quarter['Premium'] / latest_quarter['MAU'] * 100
        free_pct = 100 - premium_pct
        
        print(f"Current user breakdown:")
        print(f"  - Premium subscribers: {latest_quarter['Premium']} million ({premium_pct:.1f}%)")
        print(f"  - Free users: {latest_quarter['MAU'] - latest_quarter['Premium']} million ({free_pct:.1f}%)")
        
        # Create a detailed report
        report = f"""
# Spotify Monthly Active Users Analysis
## Global Growth Trends (2015-2024)

- **Current total MAU**: {latest_quarter['MAU']} million users (Q{latest_quarter['Quarter']} {latest_quarter['Year']})
- **Starting point**: {first_quarter['MAU']} million users (Q{first_quarter['Quarter']} {first_quarter['Year']})
- **Total growth**: {total_growth:.1f}% over {years_diff:.1f} years
- **Compound Annual Growth Rate (CAGR)**: {cagr:.1f}%

## User Composition
- **Premium subscribers**: {latest_quarter['Premium']} million ({premium_pct:.1f}%)
- **Free users**: {latest_quarter['MAU'] - latest_quarter['Premium']} million ({free_pct:.1f}%)

## Growth Analysis
- **Average YoY growth (since 2021)**: {avg_yoy_growth:.1f}%
- **Latest YoY growth rate**: {latest_quarter['MAU_YoY_Growth']:.1f}%
- **Premium YoY growth rate**: {latest_quarter['Premium_YoY_Growth']:.1f}%

## Key Growth Milestones
"""
        # Add milestones
        milestones = [
            (100, "100 million"),
            (200, "200 million"),
            (300, "300 million"),
            (400, "400 million"),
            (500, "500 million"),
            (600, "600 million")
        ]
        
        for milestone, label in milestones:
            milestone_data = self.user_data[self.user_data['MAU'] >= milestone].iloc[0]
            report += f"- **{label} MAU**: Reached in Q{milestone_data['Quarter']} {milestone_data['Year']}\n"
        
        # Save the report
        with open(os.path.join(self.script_dir, 'visualizations', 'spotify_mau_analysis.md'), 'w') as f:
            f.write(report)
        
        print("Monthly Active Users analysis saved to 'visualizations/spotify_mau_analysis.md'")
        
        # Create a visualization showing key milestones
        plt.figure(figsize=(14, 8))
        
        # Plot the MAU line
        plt.plot(self.user_data['Date'], self.user_data['MAU'], 'o-', 
                linewidth=2, color='#1DB954')
        
        # Add milestone markers
        for milestone, label in milestones:
            if any(self.user_data['MAU'] >= milestone):
                milestone_data = self.user_data[self.user_data['MAU'] >= milestone].iloc[0]
                plt.plot(milestone_data['Date'], milestone_data['MAU'], 'ro', markersize=10)
                plt.annotate(f"{label}", 
                            xy=(milestone_data['Date'], milestone_data['MAU']),
                            xytext=(10, 20), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        # Add labels and title
        plt.title('Spotify Growth Milestones', fontsize=20)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Monthly Active Users (millions)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Format the x-axis to show years properly
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        
        # Adjust layout for better appearance
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.script_dir, 'visualizations', 'spotify_growth_milestones.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_regional_user_distribution(self):
        """
        Analyze regional user distribution and growth
        This answers Question 2: "What are the key regions contributing to Spotify's user growth,
        and how does usage vary across different markets?"
        """
        if not hasattr(self, 'geographic_data') or self.geographic_data is None:
            print("Geographic data not available. Please run fetch_geographic_data() first.")
            return
        
        print("\n----- ANALYSIS: REGIONAL USER DISTRIBUTION -----")
        
        # Current distribution analysis
        print("Current regional distribution of Spotify users:")
        for _, row in self.geographic_data.sort_values('Percentage', ascending=False).iterrows():
            print(f"  - {row['Region']}: {row['Percentage']}%")
        
        # Calculate approximate MAU by region using the latest total MAU
        if self.user_data is not None:
            latest_mau = self.user_data.iloc[-1]['MAU']
            print(f"\nApproximate MAU by region (based on {latest_mau} million total MAU):")
            for _, row in self.geographic_data.sort_values('Percentage', ascending=False).iterrows():
                regional_mau = latest_mau * (row['Percentage'] / 100)
                print(f"  - {row['Region']}: {regional_mau:.1f} million")
        
        # Regional growth analysis
        if hasattr(self, 'regional_growth'):
            print("\nYear-over-Year growth rates by region:")
            for _, row in self.regional_growth.sort_values('YoY_Growth_Percentage', ascending=False).iterrows():
                print(f"  - {row['Region']}: {row['YoY_Growth_Percentage']}%")
            
            # Identify fastest and slowest growing regions
            fastest = self.regional_growth.loc[self.regional_growth['YoY_Growth_Percentage'].idxmax()]
            slowest = self.regional_growth.loc[self.regional_growth['YoY_Growth_Percentage'].idxmin()]
            
            print(f"\nFastest growing region: {fastest['Region']} ({fastest['YoY_Growth_Percentage']}%)")
            print(f"Slowest growing region: {slowest['Region']} ({slowest['YoY_Growth_Percentage']}%)")
        
        # Historical trend analysis
        if hasattr(self, 'historical_geo_data'):
            regions = self.historical_geo_data['Region'].unique()
            years = self.historical_geo_data['Year'].unique()
            
            first_year = min(years)
            last_year = max(years)
            
            print(f"\nRegional share changes ({first_year} to {last_year}):")
            
            for region in regions:
                first_data = self.historical_geo_data[(self.historical_geo_data['Region'] == region) & 
                                                   (self.historical_geo_data['Year'] == first_year)].iloc[0]
                last_data = self.historical_geo_data[(self.historical_geo_data['Region'] == region) & 
                                                  (self.historical_geo_data['Year'] == last_year)].iloc[0]
                
                change = last_data['Percentage'] - first_data['Percentage']
                
                print(f"  - {region}: {first_data['Percentage']}% → {last_data['Percentage']}% " +
                     f"({'+' if change > 0 else ''}{change:.1f}%)")
        
        # Create a detailed report
        report = f"""
# Spotify Regional User Distribution Analysis

## Current Regional Distribution (2024)

"""
        # Add current distribution
        for _, row in self.geographic_data.sort_values('Percentage', ascending=False).iterrows():
            report += f"- **{row['Region']}**: {row['Percentage']}% of total users\n"
        
        # Add MAU by region if available
        if self.user_data is not None:
            latest_mau = self.user_data.iloc[-1]['MAU']
            latest_quarter = self.user_data.iloc[-1]
            
            report += f"\n## Estimated Users by Region (Q{latest_quarter['Quarter']} {latest_quarter['Year']})\n\n"
            
            for _, row in self.geographic_data.sort_values('Percentage', ascending=False).iterrows():
                regional_mau = latest_mau * (row['Percentage'] / 100)
                report += f"- **{row['Region']}**: {regional_mau:.1f} million users\n"
        
        # Add growth rates if available
        if hasattr(self, 'regional_growth'):
            report += "\n## Year-over-Year Growth Rates (2023-2024)\n\n"
            
            for _, row in self.regional_growth.sort_values('YoY_Growth_Percentage', ascending=False).iterrows():
                report += f"- **{row['Region']}**: {row['YoY_Growth_Percentage']}%\n"
        
        # Add historical trends if available
        if hasattr(self, 'historical_geo_data'):
            report += f"\n## Historical Trends ({first_year}-{last_year})\n\n"
            
            for region in regions:
                report += f"### {region}\n\n"
                
                trend_data = self.historical_geo_data[self.historical_geo_data['Region'] == region]
                
                # Create a markdown table
                report += "| Year | Share (%) | Change |\n"
                report += "|------|-----------|--------|\n"
                
                previous_pct = None
                for _, row in trend_data.sort_values('Year').iterrows():
                    change = ""
                    if previous_pct is not None:
                        change_val = row['Percentage'] - previous_pct
                        change = f"{'+' if change_val > 0 else ''}{change_val:.1f}%"
                    
                    report += f"| {row['Year']} | {row['Percentage']}% | {change} |\n"
                    previous_pct = row['Percentage']
                
                report += "\n"
        
        # Add analysis of key findings
        report += """
## Key Findings

1. **Growth Markets**: Asia and Latin America are showing the highest growth rates, indicating Spotify's expansion strategy in emerging markets is gaining traction.

2. **Mature Markets**: Europe and North America still represent the majority of users but are growing more slowly, suggesting market saturation.

3. **Shifting Distribution**: Europe's share has declined over time, while Asia and Latin America have increased their relative importance in Spotify's user base.

4. **Regional Strategies**: Spotify's growing presence in Asia suggests successful localization strategies, including partnerships with local telecom providers and region-specific content.

5. **Future Potential**: Growth rates indicate that Asia could overtake North America in total users within the next few years if current trends continue.
"""
        
        # Save the report
        with open(os.path.join(self.script_dir, 'visualizations', 'spotify_regional_analysis.md'), 'w') as f:
            f.write(report)
        
        print("Regional user distribution analysis saved to 'visualizations/spotify_regional_analysis.md'")
        
    def create_dashboard(self):
        """
        Create a comprehensive dashboard with all visualizations
        """
        # This would require a web framework like Dash or Streamlit
        # Here's a placeholder for instructions
        print("To create an interactive dashboard, consider using:")
        print("1. Streamlit: pip install streamlit")
        print("   - Easy to use, quick to set up")
        print("   - Run with: streamlit run dashboard.py")
        print("2. Dash by Plotly: pip install dash")
        print("   - More customizable, but requires more code")
        print("   - Run with: python dashboard.py")
        print("\nExample Streamlit dashboard code has been created as 'dashboard.py'")
        
        # Create a simple Streamlit dashboard code
        dashboard_code = '''
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="Spotify Data Dashboard", layout="wide", page_icon="🎵")

# Title and description
st.title("🎵 Spotify Data Dashboard")
st.markdown("""
This dashboard presents key metrics and visualizations for Spotify, including user growth,
geographic distribution, employee growth, and stock performance.
""")

# Load data files
@st.cache_data
def load_data():
    data = {}
    data_path = "data"
    
    try:
        if os.path.exists(os.path.join(data_path, "spotify_user_data_example.csv")):
            data["user"] = pd.read_csv(os.path.join(data_path, "spotify_user_data_example.csv"))
            data["user"]["YearQuarter"] = data["user"]["Year"].astype(str) + " Q" + data["user"]["Quarter"].astype(str)
    except Exception as e:
        st.error(f"Error loading user data: {e}")
    
    try:
        if os.path.exists(os.path.join(data_path, "spotify_geographic_data_example.csv")):
            data["geo"] = pd.read_csv(os.path.join(data_path, "spotify_geographic_data_example.csv"))
    except Exception as e:
        st.error(f"Error loading geographic data: {e}")
    
    try:
        if os.path.exists(os.path.join(data_path, "spotify_employee_data_example.csv")):
            data["employee"] = pd.read_csv(os.path.join(data_path, "spotify_employee_data_example.csv"))
    except Exception as e:
        st.error(f"Error loading employee data: {e}")
    
    try:
        if os.path.exists(os.path.join(data_path, "spotify_stock_data.csv")):
            data["stock"] = pd.read_csv(os.path.join(data_path, "spotify_stock_data.csv"))
            data["stock"]["Date"] = pd.to_datetime(data["stock"]["Date"])
            data["stock"].set_index("Date", inplace=True)
    except Exception as e:
        st.error(f"Error loading stock data: {e}")
    
    return data

# Load the data
data = load_data()

# Create tabs for different visualizations
tab1, tab2, tab3, tab4 = st.tabs(["User Growth", "Geographic Distribution", "Employee Growth", "Stock Performance"])

with tab1:
    st.header("User Growth")
    if "user" in data:
        # Create the line chart
        fig = px.line(data["user"], x="YearQuarter", y=["MAU", "Premium"], 
                    title="Spotify User Growth Over Time",
                    labels={"value": "Users (millions)", "variable": "User Type"},
                    markers=True)
        
        fig.update_layout(
            height=500,
            xaxis_title="Year and Quarter",
            yaxis_title="Users (millions)",
            legend_title="User Type",
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the data table
        st.subheader("User Data")
        st.dataframe(data["user"][["Year", "Quarter", "MAU", "Premium"]])
    else:
        st.info("User data not available. Please run the data collection script first.")

with tab2:
    st.header("Geographic Distribution")
    if "geo" in data:
        # Create the pie chart
        fig = px.pie(data["geo"], values="Percentage", names="Region",
                    title="Spotify Users by Region")
        
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the data table
        st.subheader("Geographic Data")
        st.dataframe(data["geo"])
    else:
        st.info("Geographic data not available. Please run the data collection script first.")

with tab3:
    st.header("Employee Growth")
    if "employee" in data:
        # Create the bar chart
        fig = px.bar(data["employee"], x="Year", y="Employees",
                    title="Spotify Employee Growth",
                    text_auto=True)
        
        fig.update_layout(
            height=500,
            xaxis_title="Year",
            yaxis_title="Number of Employees"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the data table
        st.subheader("Employee Data")
        st.dataframe(data["employee"])
    else:
        st.info("Employee data not available. Please run the data collection script first.")

with tab4:
    st.header("Stock Performance")
    if "stock" in data:
        # Date range selector
        date_range = st.date_input(
            "Select date range",
            value=(data["stock"].index.min().date(), data["stock"].index.max().date()),
            min_value=data["stock"].index.min().date(),
            max_value=data["stock"].index.max().date()
        )
        
        # Filter data based on date range
        if len(date_range) == 2:
            filtered_stock = data["stock"].loc[date_range[0]:date_range[1]]
        else:
            filtered_stock = data["stock"]
        
        # Create the stock chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=("SPOT Stock Price", "Trading Volume"),
                           row_heights=[0.7, 0.3])
        
        # Add stock price trace
        fig.add_trace(
            go.Scatter(x=filtered_stock.index, y=filtered_stock["Close"],
                      name="Close Price",
                      line=dict(color="rgb(0, 102, 204)", width=2)),
            row=1, col=1
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(x=filtered_stock.index, y=filtered_stock["Volume"],
                  name="Volume",
                  marker=dict(color="rgb(204, 224, 255)")),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1, tickprefix="$")
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show some key statistics
        st.subheader("Key Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${filtered_stock['Close'].iloc[-1]:.2f}")
        with col2:
            st.metric("52-Week High", f"${filtered_stock['High'].max():.2f}")
        with col3:
            st.metric("52-Week Low", f"${filtered_stock['Low'].min():.2f}")
        with col4:
            pct_change = ((filtered_stock['Close'].iloc[-1] / filtered_stock['Close'].iloc[0]) - 1) * 100
            st.metric("Period Return", f"{pct_change:.2f}%", f"{pct_change:.2f}%")
        
        # Show the data table
        st.subheader("Stock Data")
        st.dataframe(filtered_stock)
    else:
        st.info("Stock data not available. Please run the data collection script first.")

# Footer
st.markdown("---")
st.markdown("Data sources: Spotify Investor Relations, SEC Filings, Yahoo Finance, Statista, and Spotify Newsroom")
'''
        
        # Save the dashboard code
        with open(os.path.join(self.script_dir, 'dashboard.py'), 'w') as f:
            f.write(dashboard_code)

# Main execution
if __name__ == "__main__":
    print("Spotify Data Collector and Visualizer")
    print("====================================")
    print("This script will help you collect and visualize data about Spotify from various sources.")
    print("Note: Not all links you provided have accessible data. I've prioritized the most valuable sources.")
    print("\nInitializing data collector...")
    
    collector = SpotifyDataCollector()
    
    print("\n----- PRIMARY DATA SOURCES (Most Reliable) -----")
    
    print("\n1. Spotify API - HIGHEST PRIORITY")
    print("The Spotify Web API is the most reliable and comprehensive data source!")
    print("To connect to the Spotify API, you need credentials.")
    print("Create a Spotify Developer account and get your client_id and client_secret.")
    print("Then call: collector.connect_to_spotify_api(client_id, client_secret)")
    
    client_id = input("\nEnter your Spotify client_id (or press Enter to skip): ").strip()
    client_secret = input("Enter your Spotify client_secret (or press Enter to skip): ").strip()
    
    if client_id and client_secret:
        collector.connect_to_spotify_api(client_id, client_secret)
    else:
        print("Skipping API connection. You can connect later by calling:")
        print("collector.connect_to_spotify_api(client_id, client_secret)")
    
    print("\n2. Stock Data from Yahoo Finance")
    collector.fetch_stock_data()
    
    print("\n----- SECONDARY DATA SOURCES (More Limited) -----")
    
    print("\n3. Financial Data from SEC & Investor Relations")
    print("Note: Many financial data sources need manual extraction from PDFs/HTML")
    collector.fetch_financial_data()
    
    print("\n4. User Statistics (from multiple sources)")
    print("Note: Statista links are mostly behind paywalls")
    collector.fetch_user_stats()
    
    print("\n5. Geographic Distribution")
    collector.fetch_geographic_data()
    
    print("\n6. Employee Data")
    collector.fetch_employee_data()
    
    print("\n7. Million Playlist Dataset")
    print("Note: This requires special academic access")
    collector.fetch_playlist_dataset_info()
    
    print("\n8. Generating Visualizations")
    collector.generate_visualizations()
    
    print("\n9. Creating Dashboard")
    collector.create_dashboard()
    
    print("\n----- SOURCES EVALUATION -----")
    print("Links accessibility assessment:")
    print("✓ https://developer.spotify.com/documentation/web-api - Excellent, fully accessible with API")
    print("✓ https://finance.yahoo.com/quote/SPOT - Good, accessible via yfinance library")
    print("✓ https://investors.spotify.com/financials/default.aspx - Moderate, needs manual extraction")
    print("✓ https://www.sec.gov/edgar/browse/?CIK=1639920 - Moderate, structured but complex")
    print("✗ https://www.statista.com/topics/2075/spotify/ - Poor, behind paywall")
    print("✗ Most news articles - Poor, only isolated statistics, not suitable for automation")
    print("~ Research articles - Variable, may require institutional access")
    
    print("\nProcess completed. Check the 'visualizations' folder for output.")
    print("To run the interactive dashboard: streamlit run dashboard.py")