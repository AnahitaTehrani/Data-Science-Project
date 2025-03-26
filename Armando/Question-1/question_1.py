import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Set global font options for a more professional look
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 22
plt.rcParams['figure.figsize'] = (11, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
# Add better spacing
plt.rcParams['figure.autolayout'] = True  

# Get the current script directory for relative file paths
script_dir = Path(__file__).parent
file_path = script_dir / 'spotify_mau.csv'
visualizations_dir = script_dir / 'visualizations'
output_pdf_path = script_dir / 'spotify_mau_analysis.pdf'
output_html_path = script_dir / 'spotify_mau_analysis.html'

# Ensure visualizations directory exists
visualizations_dir.mkdir(exist_ok=True)

# Define Spotify colors for a cohesive visual identity
spotify_green = '#1DB954'
spotify_black = '#191414'
spotify_gray = '#535353'
spotify_light_gray = '#B3B3B3'
spotify_white = '#FFFFFF'
negative_color = '#FF6B6B'

# Custom colormap for Spotify
spotify_cmap = LinearSegmentedColormap.from_list(
    'spotify', 
    [(0, '#FF6B6B'), (0.5, '#FFFFFF'), (1, spotify_green)]
)

# Load the Spotify MAU data
spotify_data = pd.read_csv(file_path)

# Calculate key metrics
latest_mau = spotify_data.iloc[0]['Users']
earliest_mau = spotify_data.iloc[-1]['Users']
total_growth = latest_mau - earliest_mau
growth_percent = (total_growth / earliest_mau) * 100

# Create a combined date column for better visualization
spotify_data['YearQuarter'] = spotify_data['Year'].astype(str) + ' ' + spotify_data['Quarter']

# Reverse the data to show chronological order
spotify_data = spotify_data.iloc[::-1].reset_index(drop=True)

# Create a date column for better plotting
spotify_data['Date'] = pd.to_datetime(spotify_data['Year'].astype(str) + 
                                      spotify_data['Quarter'].str.replace('Q', '').astype(str).apply(
                                          lambda x: f"-{int(x)*3}"))

# Calculate quarterly growth rates
spotify_data['Growth_Rate'] = spotify_data['Users'].pct_change() * 100
spotify_data['YoY_Growth'] = spotify_data['Users'].pct_change(periods=4) * 100

# Get YoY growth data for the most recent year for specific references
latest_year = spotify_data['Year'].max()
latest_year_data = spotify_data[spotify_data['Year'] == latest_year]
yoy_growth_data = []
for quarter in latest_year_data['Quarter'].unique():
    quarter_data = latest_year_data[latest_year_data['Quarter'] == quarter]
    if not quarter_data['YoY_Growth'].isna().all():
        yoy_growth_pct = quarter_data['YoY_Growth'].values[0]
        yoy_growth_data.append((quarter, yoy_growth_pct))

# Calculate average quarterly growth rate
avg_growth_rate = spotify_data['Growth_Rate'].dropna().mean()
avg_yoy_growth_rate = spotify_data['YoY_Growth'].dropna().mean()

# Find periods of highest and lowest growth
highest_growth = spotify_data['Growth_Rate'].dropna().max()
highest_growth_period = spotify_data[spotify_data['Growth_Rate'] == highest_growth]['YearQuarter'].values[0]
lowest_growth = spotify_data['Growth_Rate'].dropna().min()
lowest_growth_period = spotify_data[spotify_data['Growth_Rate'] == lowest_growth]['YearQuarter'].values[0]

# Calculate CAGR (Compound Annual Growth Rate)
years = (len(spotify_data) / 4)  # Assuming 4 quarters per year
cagr = ((latest_mau / earliest_mau) ** (1 / years) - 1) * 100

# Analyze growth phases
rolling_growth = spotify_data['Growth_Rate'].rolling(window=4).mean()
acceleration_periods = []
deceleration_periods = []

for i in range(5, len(rolling_growth)):
    if rolling_growth.iloc[i] > rolling_growth.iloc[i-4] * 1.2:  # 20% higher than year ago
        acceleration_periods.append(spotify_data['YearQuarter'].iloc[i])
    elif rolling_growth.iloc[i] < rolling_growth.iloc[i-4] * 0.8:  # 20% lower than year ago
        deceleration_periods.append(spotify_data['YearQuarter'].iloc[i])

# ------------------------- Create the visualizations ------------------------- #

# --------------- Page 1: Title and Executive Summary ---------------#
fig1 = plt.figure(figsize=(11, 8))
plt.subplots_adjust(hspace=0.5)

# Title section
plt.subplot(3, 1, 1)
plt.axis('off')
plt.text(0.5, 0.5, 'Spotify MAU Analysis', fontsize=16, ha='center')
plt.text(0.5, 0.3, f'Data from Q1 {spotify_data["Year"].min()} to Q3 {spotify_data["Year"].max()}', 
         fontsize=12, ha='center')

# Summary section
plt.subplot(3, 1, 2)
plt.axis('off')
summary_text = f"""
Current MAU: {latest_mau} million
Total Growth: +{total_growth} million
Growth %: +{growth_percent:.1f}%
CAGR: {cagr:.2f}%
Avg. Quarterly Growth: {avg_growth_rate:.2f}%
Latest YoY Growth: {yoy_growth_data[0][1]:.1f}%
"""
plt.text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center')

# Highlights section
plt.subplot(3, 1, 3)
plt.axis('off')
highlights_text = f"""
KEY INSIGHTS:
• Spotify has grown from {earliest_mau} to {latest_mau} million monthly active users since {spotify_data['Year'].min()}
• The service maintains consistent growth every quarter, with an average rate of {avg_growth_rate:.2f}%
• Strongest growth period was {highest_growth_period} at {highest_growth:.1f}%
• At current growth rates, Spotify is projected to reach 700M users by {latest_year + int((700-latest_mau)/(latest_mau*avg_growth_rate/100*4))}
"""
plt.text(0.5, 0.5, highlights_text, fontsize=12, ha='center', va='center')

# Save the figure to the visualizations folder
plt.savefig(visualizations_dir / 'executive_summary.png', dpi=300, bbox_inches='tight')
plt.close()


# --------------- Page 2: MAU Trend Analysis ---------------#
fig2 = plt.figure(figsize=(10, 8))
fig2.suptitle('Monthly Active Users Growth Trajectory', fontsize=16, fontweight='bold', color=spotify_black)

# Main user count plot - enhanced line chart with annotations
plt.subplot(3, 1, 1)
plt.plot(spotify_data['Date'], spotify_data['Users'], linewidth=3, color=spotify_green, marker='o', markersize=4)
plt.ylabel('Monthly Active Users (millions)', fontweight='bold')
plt.grid(True, alpha=0.3)
# Format y-axis to show millions with commas
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}M'))
# Annotate latest value
plt.annotate(f"{int(latest_mau)}M", 
             xy=(spotify_data['Date'].iloc[-1], spotify_data['Users'].iloc[-1]),
             xytext=(10, 10), textcoords='offset points',
             fontsize=10, fontweight='bold', color=spotify_green,
             bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=spotify_green, alpha=0.7))
# Fill area under the curve
plt.fill_between(spotify_data['Date'], 0, spotify_data['Users'], color=spotify_green, alpha=0.1)
plt.title('Spotify Monthly Active Users', fontsize=14, fontweight='bold', pad=10)
# Improve x-axis date formatting
plt.gcf().autofmt_xdate()

# YoY growth rate plot - improved bar chart with trend line
plt.subplot(3, 1, 2)
valid_yoy = spotify_data.dropna(subset=['YoY_Growth'])
# Color bars based on above/below average growth
colors = [spotify_green if x >= avg_yoy_growth_rate else negative_color for x in valid_yoy['YoY_Growth']]
bars = plt.bar(valid_yoy['Date'], valid_yoy['YoY_Growth'], color=colors, alpha=0.8, width=60)
# Add trend line
plt.plot(valid_yoy['Date'], valid_yoy['YoY_Growth'].rolling(window=4).mean(), color=spotify_black, linewidth=2)
# Add horizontal line for average
plt.axhline(y=avg_yoy_growth_rate, linestyle='--', color=spotify_gray, linewidth=1.5, 
           label=f'Avg: {avg_yoy_growth_rate:.1f}%')
plt.title('Year-over-Year Growth Rate', fontsize=14, fontweight='bold', pad=10)
plt.ylabel('Growth %', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
# Improve x-axis date formatting
plt.gcf().autofmt_xdate()

# Quarterly growth rate plot - enhanced bar chart with annotations
plt.subplot(3, 1, 3)
valid_q_growth = spotify_data.dropna(subset=['Growth_Rate'])
# Use color to distinguish positive and negative growth
colors = [spotify_green if x >= 0 else negative_color for x in valid_q_growth['Growth_Rate']]
bars = plt.bar(valid_q_growth['Date'], valid_q_growth['Growth_Rate'], width=30, color=colors, alpha=0.8)
plt.axhline(y=avg_growth_rate, linestyle='--', color=spotify_gray, linewidth=1.5, 
           label=f'Avg: {avg_growth_rate:.1f}%')
plt.title('Quarterly Growth Rate', fontsize=14, fontweight='bold', pad=10)
plt.ylabel('Growth %', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# Annotate highest and lowest growth
highest_idx = valid_q_growth['Growth_Rate'].idxmax()
lowest_idx = valid_q_growth['Growth_Rate'].idxmin()
plt.annotate(f"{valid_q_growth['Growth_Rate'].iloc[highest_idx]:.1f}%", 
             xy=(valid_q_growth['Date'].iloc[highest_idx], valid_q_growth['Growth_Rate'].iloc[highest_idx]),
             xytext=(0, 10), textcoords='offset points',
             fontsize=9, fontweight='bold', color='black', ha='center')
plt.annotate(f"{valid_q_growth['Growth_Rate'].iloc[lowest_idx]:.1f}%", 
             xy=(valid_q_growth['Date'].iloc[lowest_idx], valid_q_growth['Growth_Rate'].iloc[lowest_idx]),
             xytext=(0, -15), textcoords='offset points',
             fontsize=9, fontweight='bold', color='black', ha='center')

# Improve x-axis date formatting
plt.gcf().autofmt_xdate()

plt.tight_layout(pad=2.0)  # Add more padding between subplots
plt.savefig(visualizations_dir / 'mau_trend_analysis.png', dpi=300, bbox_inches='tight')
plt.close()


# --------------- Page 3: Growth Analysis & Projections --------------- #
fig3 = plt.figure(figsize=(10, 10))
fig3.suptitle('Growth Pattern Analysis', fontsize=14)

# 1. Rolling Average Growth Rate
plt.subplot(4, 1, 1)
growth_data = spotify_data.dropna(subset=['Growth_Rate']).copy()
growth_data['Rolling'] = growth_data['Growth_Rate'].rolling(window=4).mean()
valid_data = growth_data.dropna(subset=['Rolling'])
plt.plot(valid_data['Date'], valid_data['Rolling'], linewidth=2)
plt.title('4-Quarter Rolling Average Growth Rate', fontsize=12)
plt.ylabel('Growth Rate (%)')
plt.grid(True, alpha=0.3)

# 2. Average Quarterly Growth by Year
plt.subplot(4, 1, 2)
yearly_avg_growth = spotify_data.groupby('Year')['Growth_Rate'].mean()
plt.bar(yearly_avg_growth.index.astype(str), yearly_avg_growth.values)
plt.axhline(y=avg_growth_rate, linestyle='--', color='gray')
plt.title('Average Quarterly Growth by Year', fontsize=12)
plt.ylabel('Growth Rate (%)')
plt.grid(True, alpha=0.3, axis='y')

# 3. Average Growth by Quarter
plt.subplot(4, 1, 3)
quarterly_avg_growth = spotify_data.groupby('Quarter')['Growth_Rate'].mean()
plt.bar(quarterly_avg_growth.index, quarterly_avg_growth.values)
plt.title('Average Growth by Quarter', fontsize=12)
plt.ylabel('Growth Rate (%)')
plt.grid(True, alpha=0.3, axis='y')

# 4. Growth Projections
plt.subplot(4, 1, 4)

# Simplified projection calculation
projection_years = 5
projection_quarters = projection_years * 4
last_date = spotify_data['Date'].iloc[-1]
projection_dates = [last_date + pd.DateOffset(months=3*i) for i in range(1, projection_quarters+1)]
projected_users = [latest_mau]
quarterly_growth_factor = (1 + cagr/100) ** (1/4)

for i in range(projection_quarters):
    next_value = projected_users[-1] * quarterly_growth_factor
    projected_users.append(next_value)

# Plot historical and projected data
plt.plot(spotify_data['Date'], spotify_data['Users'], label='Historical')
plt.plot(projection_dates, projected_users[1:], linestyle='--', label='Projected')
plt.axvline(x=last_date, linestyle='--', color='gray')
plt.title(f'Projected Growth (CAGR {cagr:.2f}%)', fontsize=12)
plt.ylabel('Monthly Active Users (millions)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(visualizations_dir / 'growth_analysis_projections.png', dpi=300, bbox_inches='tight')
plt.close()


# --------------- Page 4: Detailed Metrics and Insights --------------- #
fig4 = plt.figure(figsize=(11, 8))
fig4.suptitle('Detailed Analysis & Key Insights', fontsize=16)

# Create a 2x2 grid of subplots
# Section 1: Key Performance Metrics
plt.subplot(2, 2, 1)
metrics_text = f"""
PERFORMANCE METRICS:
• Current MAU: {int(latest_mau)} million users
• Initial MAU: {int(earliest_mau)} million users
• Total Growth: +{int(total_growth)} million users
• Percent Growth: {growth_percent:.1f}%
• Time Period: {len(spotify_data) / 4:.1f} years

GROWTH RATES:
• CAGR: {cagr:.2f}%
• Avg. Quarterly Growth: {avg_growth_rate:.2f}%
• Avg. Year-over-Year Growth: {avg_yoy_growth_rate:.2f}%
• Latest Quarter Growth: {float(spotify_data['Growth_Rate'].iloc[-1]):.2f}%
• Latest YoY Growth: {float(yoy_growth_data[0][1]):.1f}%
"""
plt.text(0.5, 0.5, metrics_text, fontsize=10, ha='center', va='center', transform=plt.gca().transAxes)
plt.title('Key Performance Metrics', fontsize=12)
plt.axis('off')

# Section 2: Growth Patterns & Yearly Comparison
plt.subplot(2, 2, 3)

# Create data for the heatmap
heatmap_data = []
years = sorted(spotify_data['Year'].unique())
quarters = ['Q1', 'Q2', 'Q3', 'Q4']

for year in years:
    year_data = []
    for quarter in quarters:
        try:
            value = spotify_data[(spotify_data['Year'] == year) & (spotify_data['Quarter'] == quarter)]['Growth_Rate'].values[0]
            year_data.append(value)
        except:
            year_data.append(np.nan)  # Use NaN for missing quarters
    heatmap_data.append(year_data)

# Create enhanced heatmap with Spotify colors
cmap = LinearSegmentedColormap.from_list('spotify', [negative_color, '#FFFFFF', spotify_green])
im = plt.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=-10, vmax=10)
plt.colorbar(im, shrink=0.8, label='Quarterly Growth Rate (%)')

# Configure axes with improved formatting
plt.xticks(np.arange(len(quarters)), quarters, fontweight='bold')
plt.yticks(np.arange(len(years)), years, fontweight='bold')

# Add text annotations for all values with enhanced formatting
for i in range(len(years)):
    for j in range(len(quarters)):
        if not np.isnan(heatmap_data[i][j]):
            value = heatmap_data[i][j]
            color = "black" if abs(value) < 7 else "white"
            weight = 'bold' if abs(value) > 5 else 'normal'
            plt.text(j, i, f"{value:.1f}%",
                   ha="center", va="center", color=color,
                   fontsize=9, fontweight=weight)

plt.title('Quarterly Growth Heatmap', fontsize=12, fontweight='bold')

# Section 3: Key Insights
plt.subplot(2, 2, 4)

# Calculate additional insights
consecutive_growth = sum(1 for x in spotify_data['Growth_Rate'].dropna() if x > 0)
total_periods = len(spotify_data['Growth_Rate'].dropna())
growth_consistency = consecutive_growth / total_periods * 100

insights_text = f"""
KEY INSIGHTS:
• Spotify maintained positive growth in {consecutive_growth} of {total_periods} quarters ({growth_consistency:.1f}%)
• Growth acceleration periods: {', '.join(acceleration_periods[:2]) if acceleration_periods else 'N/A'}
• Growth deceleration periods: {', '.join(deceleration_periods[:2]) if deceleration_periods else 'N/A'}
• Best performing quarter: {quarterly_avg_growth.idxmax()} ({quarterly_avg_growth.max():.1f}%)
• Best performing year: {yearly_avg_growth.idxmax()} ({yearly_avg_growth.max():.1f}%)
• Projected to reach 1B users by ~{latest_year + int(np.log(1000/latest_mau) / np.log(1 + cagr/100))}
"""

plt.text(0.5, 0.5, insights_text, fontsize=10, ha='center', va='center', transform=plt.gca().transAxes)
plt.title('Growth Insights', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig(visualizations_dir / 'detailed_metrics_insights.png', dpi=300, bbox_inches='tight')
plt.close()


# --------------- individual visualizations ------------------------------ # 

# Also save individual visualizations for more granular reporting

# Save individual visualizations - simplified
print(f"Creating simplified visualizations at: {visualizations_dir}")

# 1. MAU Growth Line Chart
plt.figure()
plt.plot(spotify_data['Date'], spotify_data['Users'])
plt.title('Spotify Monthly Active Users Growth')
plt.ylabel('Monthly Active Users (millions)')
plt.grid(True)
plt.savefig(visualizations_dir / 'mau_growth_line.png')
plt.close()

# 2. Year-over-Year Growth
plt.figure(figsize=(10, 6))
valid_yoy = spotify_data.dropna(subset=['YoY_Growth'])
# Color bars based on above/below average growth
colors = [spotify_green if x >= avg_yoy_growth_rate else negative_color for x in valid_yoy['YoY_Growth']]
bars = plt.bar(valid_yoy['Date'], valid_yoy['YoY_Growth'], color=colors, alpha=0.8, width=60)
# Add average line with better formatting
plt.axhline(y=avg_yoy_growth_rate, linestyle='--', color=spotify_gray, linewidth=1.5, 
           label=f'Avg: {avg_yoy_growth_rate:.1f}%')
# Improve formatting and labels
plt.title('Spotify Year-over-Year User Growth', fontsize=14, fontweight='bold')
plt.ylabel('Growth Rate (%)', fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.legend(loc='upper right')

# Add annotations for highest and lowest points
highest_yoy = valid_yoy['YoY_Growth'].max()
highest_yoy_idx = valid_yoy['YoY_Growth'].idxmax()
lowest_yoy = valid_yoy['YoY_Growth'].min()
lowest_yoy_idx = valid_yoy['YoY_Growth'].idxmin()

plt.annotate(f"{highest_yoy:.1f}%", 
             xy=(valid_yoy['Date'].iloc[highest_yoy_idx-valid_yoy.index[0]], highest_yoy),
             xytext=(0, 10), textcoords='offset points',
             fontsize=9, fontweight='bold', ha='center')
             
plt.annotate(f"{lowest_yoy:.1f}%", 
             xy=(valid_yoy['Date'].iloc[lowest_yoy_idx-valid_yoy.index[0]], lowest_yoy),
             xytext=(0, -15), textcoords='offset points',
             fontsize=9, fontweight='bold', ha='center')

# Improve x-axis formatting
plt.gcf().autofmt_xdate()
# Add a light background to highlight the data
plt.gca().set_facecolor('#f8f8f8')
plt.savefig(visualizations_dir / 'yoy_growth_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Quarterly Growth Heatmap
plt.figure(figsize=(10, 6))
cmap = LinearSegmentedColormap.from_list('spotify', [negative_color, '#FFFFFF', spotify_green])
plt.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=-10, vmax=10)
plt.colorbar(label='Quarterly Growth Rate (%)')
plt.xticks(np.arange(len(quarters)), quarters, fontweight='bold')
plt.yticks(np.arange(len(years)), years, fontweight='bold')
# Add text annotations
for i in range(len(years)):
    for j in range(len(quarters)):
        if not np.isnan(heatmap_data[i][j]):
            value = heatmap_data[i][j]
            color = "black" if abs(value) < 7 else "white"
            plt.text(j, i, f"{value:.1f}%", ha="center", va="center", color=color, fontsize=9)
plt.title('Quarterly Growth Heatmap by Year', fontsize=14, fontweight='bold')
plt.savefig(visualizations_dir / 'quarterly_growth_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Growth Projections
plt.figure()
plt.plot(spotify_data['Date'], spotify_data['Users'], label='Historical')
plt.plot(projection_dates, projected_users[1:], linestyle='--', label='Projected')
plt.axvline(x=last_date, linestyle='--', color='gray')
plt.title(f'Projected Growth (CAGR {cagr:.2f}%)')
plt.ylabel('Monthly Active Users (millions)')
plt.legend()
plt.grid(True)
plt.savefig(visualizations_dir / 'growth_projections.png')
plt.close()

print(f"Individual visualizations saved to: {visualizations_dir}")

# Save all visualizations as a single PDF file - simplified
print(f"Creating PDF report using existing high-quality visualizations: {output_pdf_path}")
with PdfPages(output_pdf_path) as pdf:
    # Use our previously created high-quality visualizations
    image_files = [
        'executive_summary.png',
        'mau_trend_analysis.png',
        'growth_analysis_projections.png',
        'detailed_metrics_insights.png'
    ]
    
    # Add each visualization to the PDF
    for img_file in image_files:
        img_path = visualizations_dir / img_file
        # Open the image file
        img = plt.imread(img_path)
        # Create a figure with the image's aspect ratio
        height, width, _ = img.shape
        fig = plt.figure(figsize=(11, 11 * height / width))
        # Display the image without axes
        plt.imshow(img)
        plt.axis('off')
        # Save to PDF
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)

print(f"High-quality PDF report created at: {output_pdf_path}")

# Create an HTML version with interactive visualizations (optional)
print(f"Creating HTML report at: {output_html_path}")
with open(output_html_path, 'w') as html_file:
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Spotify MAU Analysis</title>
        <style>
            body {{
                font-family: 'Helvetica', sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f8f8;
            }}
            h1, h2 {{
                color: #1DB954;  /* Spotify green */
            }}
            .visualization {{
                margin: 30px 0;
                text-align: center;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <h1>Spotify MAU Analysis</h1>
        <p>Data from Q1 {spotify_data["Year"].min()} to Q3 {spotify_data["Year"].max()}</p>
        
        <div class="visualization">
            <h2>Executive Summary</h2>
            <img src="visualizations/executive_summary.png" alt="Executive Summary">
        </div>
        
        <div class="visualization">
            <h2>MAU Trend Analysis</h2>
            <img src="visualizations/mau_trend_analysis.png" alt="MAU Trend Analysis">
        </div>
        
        <div class="visualization">
            <h2>Growth Analysis & Projections</h2>
            <img src="visualizations/growth_analysis_projections.png" alt="Growth Analysis">
        </div>
        
        <div class="visualization">
            <h2>Detailed Metrics and Insights</h2>
            <img src="visualizations/detailed_metrics_insights.png" alt="Detailed Metrics">
        </div>
    </body>
    </html>
    """
    html_file.write(html_content)

print(f"HTML report created at: {output_html_path}")
