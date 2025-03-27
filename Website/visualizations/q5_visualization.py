# q5_visualization.py - Visualizations for Question 5: Demographic Factors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_age_demographics_visualization():
    """
    Creates a bar chart showing Spotify's age demographics.
    Includes interactive dropdown to select specific age groups.
    """
    # Data
    age_groups = ['18-24', '25-34', '35-44', '45-54', '+55']
    percentages = [26, 29, 16, 11, 19]
    colors = ['#ee3b3b', '#00ff00', '#ffff00', '#ee1289', '#ba55d3']

    # Create figure
    fig = make_subplots(rows=1, cols=1)

    # Add traces
    for i in range(len(age_groups)):
        fig.add_trace(go.Bar(
            x=[age_groups[i]],  # The specific age group
            y=[percentages[i]],  # The percentage
            name=age_groups[i],  # Legend name
            marker=dict(color=colors[i]),  # Custom color
        ))

    # Layout customization
    fig.update_layout(
        title="Spotify's Age Demographics in (%)",
        xaxis_title="Age Groups",
        yaxis_title="Percentage (%)",
        barmode='group',  # Group bars
        showlegend=True,  # Show legend
        legend=dict(
            title="Age Groups",
            orientation="v",  # Vertical legend
            x=1.05,  # Position
            y=1
        ),
        updatemenus=[dict(
            buttons=[
                dict(label="Show 18-24", method="restyle", args=[{"visible": [True, False, False, False, False]}]),
                dict(label="Show 25-34", method="restyle", args=[{"visible": [False, True, False, False, False]}]),
                dict(label="Show 35-44", method="restyle", args=[{"visible": [False, False, True, False, False]}]),
                dict(label="Show 45-54", method="restyle", args=[{"visible": [False, False, False, True, False]}]),
                dict(label="Show +55", method="restyle", args=[{"visible": [False, False, False, False, True]}]),
                dict(label="Show All", method="restyle", args=[{"visible": [True, True, True, True, True]}])
            ],
            direction="down",
            showactive=True,
        )],
        plot_bgcolor='white',  # White background
        paper_bgcolor='white',
    )
    
    return fig

def create_gender_demographics_visualization():
    """
    Creates a pie chart showing Spotify's gender demographics.
    """
    # Sample gender data (you can replace with actual data)
    labels = ['Male', 'Female', 'Other/Not Specified']
    values = [52, 45, 3]  # Percentages
    colors = ['#1DB954', '#FF1493', '#4169E1']  # Spotify green, pink, blue
    
    # Create the figure
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,  # Creates a donut chart
        marker=dict(colors=colors)
    )])
    
    # Update layout
    fig.update_layout(
        title='Spotify Users by Gender (%)',
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    return fig