# app.py - Main application entry point
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os

# Import the visualization function directly
from visualizations.q1_visualization import create_user_growth_visualization

# Initialize the Dash app
app = dash.Dash(__name__, 
                suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

# Required for Render deployment
server = app.server

# App layout with navigation and content div
app.layout = html.Div([
    # Store the current page
    dcc.Location(id='url', refresh=False),
    
    # Header with navigation
    html.Div([
        html.H1("Spotify Insights: User Behavior, Media Influence, and Market Dynamics", className='header-title'),
        html.Div([
            dcc.Link('Home', href='/', className='nav-link'),
            dcc.Link('User Growth', href='/question1', className='nav-link'),
            dcc.Link('Media Sentiment', href='/question2', className='nav-link'),
            dcc.Link('Regional Activity', href='/question3', className='nav-link'),
            dcc.Link('Free vs Premium', href='/question4', className='nav-link'),
            dcc.Link('Demographics', href='/question5', className='nav-link'),
            dcc.Link('Chart Comparison', href='/question6', className='nav-link'),
            dcc.Link('Price Impact', href='/question7', className='nav-link'),
            dcc.Link('Track Popularity', href='/question8', className='nav-link'),
            dcc.Link('Imprint', href='/imprint', className='nav-link'),
        ], className='nav-links')
    ], className='header'),
    
    # Content div - will be populated based on URL
    html.Div(id='page-content', className='content')
])

# Home page layout
def get_home_layout():
    return html.Div([
        html.H2("Welcome to Spotify Insights: User Behavior, Media Influence, and Market Dynamics"),
        html.P("This interactive dashboard presents visualizations for eight research questions about Spotify's growth, user behavior, and market dynamics."),
        html.P("Use the navigation bar above to explore each research question and its corresponding visualizations."),
        
        html.Div([
            html.H3("Research Questions Overview"),
            html.Ul([
                html.Li([html.A("User Growth", href="/question1"), 
                         " - How many users worldwide use Spotify monthly, and how has this number changed over time?"]),
                html.Li([html.A("Media Sentiment", href="/question2"), 
                         " - What does the sentiment analysis of news articles about Spotify reveal about the relationship between media coverage and stock performance?"]),
                html.Li([html.A("Regional Activity", href="/question3"), 
                         " - Which global regions show the highest Spotify streaming activity, and how has this changed over time?"]),
                html.Li([html.A("Free vs Premium", href="/question4"), 
                         " - How does the ratio of free users (Free with ads) to premium subscribers impact Spotify's revenue?"]),
                html.Li([html.A("Demographics", href="/question5"), 
                         " - What demographic factors (e.g., age, gender) influence Spotify's user base and subscription trends?"]),
                html.Li([html.A("Chart Comparison", href="/question6"), 
                         " - How do the popular music charts in general compare to the popular Spotify music charts?"]),
                html.Li([html.A("Price Impact", href="/question7"), 
                         " - How do price changes affect Spotify's growth?"]),
                html.Li([html.A("Track Popularity", href="/question8"), 
                         " - Is there a tendency for popular tracks (Track Table) to appear more frequently in playlists (Playlist-Track Table)?"]),
            ])
        ], className='home-overview')
    ])

# Template for a research question page (specifically for Question 1 initially)
def get_research_question_layout(question_number):
    questions = {
        1: "How many users worldwide use Spotify monthly, and how has this number changed over time?",
        2: "What does the sentiment analysis of news articles about Spotify reveal about the relationship between media coverage and stock performance?",
        3: "Which global regions show the highest Spotify streaming activity, and how has this changed over time?",
        4: "How does the ratio of free users (Free with ads) to premium subscribers impact Spotify's revenue?",
        5: "What demographic factors (e.g., age, gender) influence Spotify's user base and subscription trends?",
        6: "How do the popular music charts in general compare to the popular Spotify music charts?",
        7: "How do price changes affect Spotify's growth?",
        8: "Is there a tendency for popular tracks (Track Table) to appear more frequently in playlists (Playlist-Track Table)?"
    }
    
    descriptions = {
        1: "This visualization tracks Spotify's monthly active users (MAU) worldwide over time, showing how the platform has grown since its launch.",
        2: "This analysis examines the correlation between media sentiment in news articles about Spotify and the company's stock performance.",
        3: "This visualization shows which global regions have the highest Spotify streaming activity and how these patterns have evolved over time.",
        4: "This analysis explores how the ratio between free and premium users affects Spotify's revenue streams and overall financial performance.",
        5: "This visualization breaks down Spotify's user base by demographic factors to identify patterns in user behavior and subscription preferences.",
        6: "This comparison examines the differences and similarities between popular music charts in general and Spotify's own music charts.",
        7: "This analysis investigates how Spotify's price changes have impacted user growth and subscription rates over time.",
        8: "This visualization explores whether popular tracks tend to appear more frequently in user-created playlists."
    }
    
    if question_number == 1:
        # Special handling for Question 1 to get it working first
        try:
            csv_path = 'Armando/Question-1/spotify_users.csv'
            df = pd.read_csv(csv_path)
            
            # Get unique years for the dropdown
            years = sorted(df['Year'].unique())
            
            return html.Div([
                html.H2(f"Research Question {question_number}"),
                html.P(questions[question_number], className="research-question"),
                html.P(descriptions[question_number]),
                
                # Add the dropdown
                html.Div([
                    html.Label("Select Year:"),
                    dcc.Dropdown(
                        id='dropdown-q1',
                        options=[{'label': 'All Years', 'value': 'all'}] + 
                                [{'label': str(year), 'value': year} for year in years],
                        value='all',
                        clearable=False
                    )
                ], className='dropdown-container'),
                
                # Visualization container
                html.Div([
                    dcc.Graph(id='graph-q1', figure=create_user_growth_visualization(df, selected_year='all'))
                ], className='viz-container')
            ])
        except Exception as e:
            return html.Div([
                html.H2(f"Research Question {question_number}"),
                html.P(questions[question_number], className="research-question"),
                html.P(descriptions[question_number]),
                html.P(f"Error loading visualization: {str(e)}", style={'color': 'red'})
            ])
    else:
        # Generic template for other questions (can be expanded later)
        return html.Div([
            html.H2(f"Research Question {question_number}"),
            html.P(questions[question_number], className="research-question"),
            html.P(descriptions[question_number]),
            html.P("This visualization is coming soon.")
        ])

# Imprint page layout
def get_imprint_layout():
    return html.Div([
        html.H2("Imprint"),
        html.P("Your Name or University Name"),
        html.P("Contact Information: your.email@example.com"),
    ])

# Callback to update page content based on URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/':
        return get_home_layout()
    elif pathname == '/question1':
        return get_research_question_layout(1)
    elif pathname == '/question2':
        return get_research_question_layout(2)
    elif pathname == '/question3':
        return get_research_question_layout(3)
    elif pathname == '/question4':
        return get_research_question_layout(4)
    elif pathname == '/question5':
        return get_research_question_layout(5)
    elif pathname == '/question6':
        return get_research_question_layout(6)
    elif pathname == '/question7':
        return get_research_question_layout(7)
    elif pathname == '/question8':
        return get_research_question_layout(8)
    elif pathname == '/imprint':
        return get_imprint_layout()
    else:
        return get_home_layout()  # Default to home page

# Define Spotify brand colors
spotify_green = '#1DB954'
light_bg = '#FFFFFF'
dark_text = '#191414'
spotify_colors = ['#1DB954', '#1A73E8', '#191414', '#535353', '#B3B3B3']

# Callback for research question 1 - User growth visualization
@app.callback(
    Output('graph-q1', 'figure'),
    [Input('dropdown-q1', 'value')]
)
def update_graph_q1(selected_year):
    try:
        # Adjust this path based on where your data is stored
        csv_path = 'Armando/Question-1/spotify_users.csv'
        df = pd.read_csv(csv_path)
        
        # Create the visualization using the function from q1_visualization.py
        fig = create_user_growth_visualization(df, selected_year)
        
        return fig
    except Exception as e:
        # Return an empty figure with error message
        return {
            'data': [],
            'layout': {
                'title': f"Error loading visualization: {str(e)}",
                'height': 500
            }
        }

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)