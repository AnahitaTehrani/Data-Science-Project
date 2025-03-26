# app.py - Main application entry point
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from visualizations.q1_visualization import create_user_growth_visualization
# Import other visualization functions as needed

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

# Sample data for demo visualizations
def get_sample_data():
    # This function returns sample data that will be replaced with your actual data
    return pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D', 'E', 'F'],
        'Values': [4, 7, 2, 5, 9, 3],
        'Year': [2019, 2020, 2021, 2019, 2020, 2021]
    })

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

# Template for a research question page
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
    
    # needs to be updated once visualizations are added
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
    
    return html.Div([
        html.H2(f"Research Question {question_number}"),
        html.P(questions[question_number], className="metric-value"),
        html.P(descriptions[question_number]),
        
        # Example of an interactive visualization with dropdown
        html.Div([
            dcc.Graph(id=f'graph-q{question_number}')
        ], className='viz-container')
    ])

# Imprint page layout
def get_imprint_layout():
    return html.Div([
        html.H2("Imprint"),
        html.P("Your Name or University Name"),
        html.P("Contact Information: your.email@example.com"),
        # Add more details as needed
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

# Light mode theme colors for visualizations
spotify_green = '#1DB954'
light_bg = '#FFFFFF'
dark_text = '#191414'
spotify_colors = ['#1DB954', '#1A73E8', '#191414', '#535353', '#B3B3B3']

# Callback for research question 1 - Line chart for user growth over time
@app.callback(
    Output('graph-q1', 'figure'),
    [Input('dropdown-q1', 'value')]
)
def update_graph_q1(selected_year):
    # Load your actual data here
    # df = pd.read_csv('path_to_your_data.csv')
    
    # For now, using sample data
    df = get_sample_data()
    
    # Use your visualization function
    return create_user_growth_visualization(df, selected_year)

# Callback for research question 2 - Scatter plot for sentiment vs stock performance
@app.callback(
    Output('graph-q2', 'figure'),
    [Input('dropdown-q2', 'value')]
)
def update_graph_q2(selected_year):
    # Get sample data (to be replaced with actual data)
    df = get_sample_data()
    
    # Filter data based on dropdown selection
    if selected_year != 'all':
        filtered_df = df[df['Year'] == selected_year]
    else:
        filtered_df = df
    
    # Create a scatter plot for sentiment vs stock performance
    fig = px.scatter(
        filtered_df, 
        x='Category', 
        y='Values',
        size='Values',
        title=f'Media Sentiment vs Stock Performance ({selected_year if selected_year != "all" else "All Years"})',
        color='Values',
        color_continuous_scale=['#B3B3B3', '#1DB954']  # Light gray to Spotify green
    )
    
    fig.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Stock Price ($)",
        plot_bgcolor=light_bg,
        paper_bgcolor=light_bg,
        font_color=dark_text
    )
    
    return fig

# Callback for research question 3 - Bar chart for regional streaming activity
@app.callback(
    Output('graph-q3', 'figure'),
    [Input('dropdown-q3', 'value')]
)
def update_graph_q3(selected_year):
    # Get sample data (to be replaced with actual data)
    df = get_sample_data()
    
    # Filter data based on dropdown selection
    if selected_year != 'all':
        filtered_df = df[df['Year'] == selected_year]
    else:
        filtered_df = df
    
    # Create a bar chart for regional streaming activity
    fig = px.bar(
        filtered_df, 
        x='Category', 
        y='Values',
        title=f'Regional Streaming Activity ({selected_year if selected_year != "all" else "All Years"})',
        color_discrete_sequence=spotify_colors
    )
    
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Streaming Hours (millions)",
        plot_bgcolor=light_bg,
        paper_bgcolor=light_bg,
        font_color=dark_text
    )
    
    return fig

# Callback for research question 4 - Pie chart for free vs premium users
@app.callback(
    Output('graph-q4', 'figure'),
    [Input('dropdown-q4', 'value')]
)
def update_graph_q4(selected_year):
    # Get sample data (to be replaced with actual data)
    df = get_sample_data()
    
    # Filter data based on dropdown selection
    if selected_year != 'all':
        filtered_df = df[df['Year'] == selected_year]
    else:
        filtered_df = df
    
    # Create a pie chart for free vs premium users
    fig = px.pie(
        filtered_df, 
        values='Values', 
        names='Category',
        title=f'Free vs Premium Users and Revenue Impact ({selected_year if selected_year != "all" else "All Years"})',
        color_discrete_sequence=[spotify_green, '#1A73E8', '#535353', '#B3B3B3']
    )
    
    fig.update_layout(
        plot_bgcolor=light_bg,
        paper_bgcolor=light_bg,
        font_color=dark_text
    )
    
    return fig

# Callback for research question 5 - Bar chart for demographic factors
@app.callback(
    Output('graph-q5', 'figure'),
    [Input('dropdown-q5', 'value')]
)
def update_graph_q5(selected_year):
    # Get sample data (to be replaced with actual data)
    df = get_sample_data()
    
    # Filter data based on dropdown selection
    if selected_year != 'all':
        filtered_df = df[df['Year'] == selected_year]
    else:
        filtered_df = df
    
    # Create a grouped bar chart for demographic breakdown
    fig = px.bar(
        filtered_df, 
        x='Category', 
        y='Values',
        title=f'User Demographics ({selected_year if selected_year != "all" else "All Years"})',
        color_discrete_sequence=spotify_colors,
        barmode='group'
    )
    
    fig.update_layout(
        xaxis_title="Age Group",
        yaxis_title="Number of Users (millions)",
        plot_bgcolor=light_bg,
        paper_bgcolor=light_bg,
        font_color=dark_text
    )
    
    return fig

# Callback for research question 6 - Dual axis chart for chart comparison
@app.callback(
    Output('graph-q6', 'figure'),
    [Input('dropdown-q6', 'value')]
)
def update_graph_q6(selected_year):
    # Get sample data (to be replaced with actual data)
    df = get_sample_data()
    
    # Filter data based on dropdown selection
    if selected_year != 'all':
        filtered_df = df[df['Year'] == selected_year]
    else:
        filtered_df = df
    
    # Create a bar chart for chart comparison (would be replaced with dual-axes)
    fig = px.bar(
        filtered_df, 
        x='Category', 
        y='Values',
        title=f'Popular Music Charts vs Spotify Charts ({selected_year if selected_year != "all" else "All Years"})',
        color='Year',
        color_discrete_sequence=[spotify_green, '#1A73E8', '#535353']
    )
    
    fig.update_layout(
        xaxis_title="Track",
        yaxis_title="Ranking Position",
        plot_bgcolor=light_bg,
        paper_bgcolor=light_bg,
        font_color=dark_text
    )
    
    return fig

# Callback for research question 7 - Line chart for price impact
@app.callback(
    Output('graph-q7', 'figure'),
    [Input('dropdown-q7', 'value')]
)
def update_graph_q7(selected_year):
    # Get sample data (to be replaced with actual data)
    df = get_sample_data()
    
    # Filter data based on dropdown selection
    if selected_year != 'all':
        filtered_df = df[df['Year'] == selected_year]
    else:
        filtered_df = df
    
    # Create a line chart for price impact on growth
    fig = px.line(
        filtered_df, 
        x='Category', 
        y='Values',
        title=f'Price Changes and Growth Impact ({selected_year if selected_year != "all" else "All Years"})',
        markers=True,
        color_discrete_sequence=[spotify_green]
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Subscription Growth Rate (%)",
        plot_bgcolor=light_bg,
        paper_bgcolor=light_bg,
        font_color=dark_text
    )
    
    # Add annotations for price change events
    fig.add_annotation(
        x="C",
        y=2,
        text="Price Increase",
        showarrow=True,
        arrowhead=1,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=spotify_green
    )
    
    return fig

# Callback for research question 8 - Scatter plot for track popularity vs playlist frequency
@app.callback(
    Output('graph-q8', 'figure'),
    [Input('dropdown-q8', 'value')]
)
def update_graph_q8(selected_year):
    # Get sample data (to be replaced with actual data)
    df = get_sample_data()
    
    # Filter data based on dropdown selection
    if selected_year != 'all':
        filtered_df = df[df['Year'] == selected_year]
    else:
        filtered_df = df
    
    # Create a scatter plot for track popularity vs playlist frequency
    fig = px.scatter(
        filtered_df, 
        x='Category', 
        y='Values',
        size='Values',
        title=f'Track Popularity vs Playlist Frequency ({selected_year if selected_year != "all" else "All Years"})',
        color_discrete_sequence=spotify_colors
    )
    
    fig.update_layout(
        xaxis_title="Track Popularity Score",
        yaxis_title="Playlist Appearances",
        plot_bgcolor=light_bg,
        paper_bgcolor=light_bg,
        font_color=dark_text
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)