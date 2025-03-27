import plotly.graph_objects as go
import pandas as pd
import dash
from dash import dcc, html
import os

# Create a simple app with forced CSS rendering
app = dash.Dash(
    __name__,
    assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
)

# Create assets directory if it doesn't exist
assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

# Create custom CSS to force SVG visibility
custom_css = """
.js-plotly-plot, .plot-container, .svg-container, svg {
    visibility: visible !important;
    opacity: 1 !important;
    display: block !important;
}
"""
with open(os.path.join(assets_dir, 'custom.css'), 'w') as f:
    f.write(custom_css)

# Get the correct path to the data file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(current_dir), 'Armando', 'Question-1', 'spotify_users.csv')

# Try to read your data file
df = pd.read_csv(data_path)

# Create a simple figure
fig = go.Figure(
    data=[
        go.Scatter(
            x=df['Date'], 
            y=df['Monthly Active Users (Millions)'],
            mode='lines+markers',
            name='MAU',
            line=dict(color='#FF0000', width=5),  # Bright red for visibility
            marker=dict(size=12, color='#FF0000', line=dict(width=2, color='white'))
        )
    ]
)

# Update the layout to make the graph more visible
fig.update_layout(
    title={
        'text': 'Spotify Monthly Active Users',
        'font': {'size': 24, 'color': 'black'}
    },
    xaxis_title={'text': 'Date', 'font': {'size': 18, 'color': 'black'}},
    yaxis_title={'text': 'Monthly Active Users (Millions)', 'font': {'size': 18, 'color': 'black'}},
    height=600,  # Make the graph taller
    width=1000,  # Make the graph wider
    template='plotly_white',  # White background for better visibility
    plot_bgcolor='white',   
    paper_bgcolor='white',  
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(0, 0, 0, 0.2)',
        tickfont=dict(color='black', size=14)
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(0, 0, 0, 0.2)',
        tickfont=dict(color='black', size=14)
    )
)

# Super simple layout with just the graph
app.layout = html.Div([
    html.H1('Spotify User Growth Analysis', 
           style={
               'textAlign': 'center', 
               'margin': '20px',
               'color': 'black',
               'padding': '20px'
           }),
    html.Div([
        dcc.Graph(
            id='test-graph', 
            figure=fig,
            style={'height': '700px', 'width': '100%'}
        )
    ], style={
        'padding': '20px',
        'borderRadius': '5px',
        'border': '1px solid #ddd'
    })
], style={
    'maxWidth': '1200px',
    'margin': '0 auto',
    'padding': '20px',
    'fontFamily': 'Arial, sans-serif',
    'backgroundColor': 'white'
})

# Force full HTML with required scripts
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Spotify Users Dashboard</title>
        {%css%}
        <style>
            body, html {
                margin: 0;
                padding: 0;
                background-color: white;
            }
            .js-plotly-plot, .plot-container, .svg-container {
                visibility: visible !important;
                opacity: 1 !important;
            }
        </style>
    </head>
    <body>
        <div id="react-entry-point">
            <div class="_dash-loading">
                Loading...
            </div>
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Run with specific host and port
if __name__ == '__main__':
    print("Dashboard running at http://127.0.0.1:8050/")
    print("If you don't see the graph, try opening this URL in a different browser")
    app.run_server(debug=True, host='127.0.0.1', port=8050)