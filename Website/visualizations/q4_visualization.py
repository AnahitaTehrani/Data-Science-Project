# q4_visualization.py - Visualizations for Question 4: Free vs Premium Users
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def create_revenue_visualization():
    """
    Creates a bar chart showing revenue from ad-supported users, premium subscribers, and total Spotify revenue.
    Includes interactive dropdown to toggle between different revenue types.
    """
    # Data converted from Million to Billion EUR
    data = {
        'Year': [2019, 2020, 2021, 2022, 2023, 2024],
        'Ad-Supported': [678 / 1000, 745 / 1000, 1.21, 1.48, 1.68, 1.85],
        'Premium Subscribers': [6.09, 7.14, 8.46, 10.25, 11.57, 13.82],
        'Spotify Total': [6.76, 7.88, 9.67, 11.73, 13.25, 15.67]
    }

    df = pd.DataFrame(data)

    # Define custom colors
    custom_colors = {
        "Ad-Supported": "#40e0d0",
        "Premium Subscribers": "#ff1493",
        "Spotify Total": "#b452cd"
    }

    # Create the plot
    fig = go.Figure()

    # Add traces for each revenue type
    for revenue_type in ['Ad-Supported', 'Premium Subscribers', 'Spotify Total']:
        fig.add_trace(go.Bar(
            x=df['Year'],
            y=df[revenue_type],
            name=revenue_type,
            marker=dict(color=custom_colors[revenue_type]),
            visible=True  # Initially all visible
        ))

    # Revenue Selection / Dropdown menu 
    fig.update_layout(
        title="Types of Spotify Revenue (2019-2024) (in Billion EUR)",
        barmode='group',
        xaxis_title="Year",
        yaxis_title="Revenue (Billion EUR)",
        template="plotly_white",  # White background for better contrast
        legend_title_text="Revenue",
        updatemenus=[dict(
            buttons=[
                dict(
                    label="Show Ad-Supported",
                    method="restyle",
                    args=[{"visible": [True, False, False]}]  # Show Ad-Supported
                ),
                dict(
                    label="Show Premium Subscribers",
                    method="restyle",
                    args=[{"visible": [False, True, False]}]  # Show Premium Subscribers
                ),
                dict(
                    label="Show Spotify Total",
                    method="restyle",
                    args=[{"visible": [False, False, True]}]  # Show Spotify Total
                ),
                dict(
                    label="Show All",
                    method="restyle",
                    args=[{"visible": [True, True, True]}]  # Show all
                )
            ],
            direction="down",
            showactive=True,
        )],
    )
    
    return fig

def create_users_comparison_visualization():
    """
    Creates a line chart comparing ad-supported users vs. premium subscribers over time.
    """
    # Create the data
    data = {
        "Year": [2019, 2020, 2021, 2022, 2023, 2024],
        "Ad-Supported Users": [546, 717, 874, 1076, 1400, 1608],
        "Premium Subscribers": [445, 567, 675, 770, 892, 1000]
    }

    df = pd.DataFrame(data)

    # Define custom colors
    custom_colors = {
        "Ad-Supported Users": "#00bfff",  
        "Premium Subscribers": "#ff1493"  
    }

    # Create line chart
    fig = px.line(df, x="Year", y=["Ad-Supported Users", "Premium Subscribers"], 
                markers=True, 
                title="Ad-Supported Users vs. Premium Subscribers (2019-2024)",
                labels={"value": "Numbers in Million", "variable": "Legend"},
                color_discrete_map=custom_colors)  # Assign custom colors

    # Add Spotify green theme
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color='#191414',
        title_font_color='#191414',
        legend_title_font_color='#191414',
        legend_title_text="User Type"
    )
    
    return fig