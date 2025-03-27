# app.py - Main application entry point
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
from visualizations.q2_visualization import load_and_preprocess_data, create_visualization
from visualizations.q4_visualization import create_revenue_visualization, create_users_comparison_visualization
from visualizations.q5_visualization import create_age_demographics_visualization, create_gender_demographics_visualization
from visualizations.q1_visualization import create_user_growth_visualization
from visualizations.q3_visualization import render_question3, update_map
from visualizations.q6_visualization import render_question6

app = dash.Dash(__name__, 
                suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

# Add this for Font Awesome icons
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Spotify Insights</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Required for Render deployment
server = app.server

# App layout with navigation and content div
app.layout = html.Div([
    # Store the current page
    dcc.Location(id='url', refresh=False),
    
    # Header with navigation
    html.Div([
        html.H1("Spotify’s User Base: Stats, Growth & Trends", className='header-title'),
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
        html.P("Hi, we are a group of four students from CAU Kiel working on a Data Science project focused on Spotify."),
        html.P("Spotify is a leading global music streaming platform with millions of monthly active users. This analysis examines the platform’s user growth over time, the impact of economic factors, and regional usage trends. Furthermore, it explores the balance between free and premium subscribers and the effects of pricing changes. Additionally, the study investigates trends in Spotify’s music charts to provide deeper insights into user behavior. "),
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
            csv_path = 'data/spotify_users.csv'
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
                    dcc.Graph(id='graph-q1')
                ], className='viz-container')
            ])
        except Exception as e:
            return html.Div([
                html.H2(f"Research Question {question_number}"),
                html.P(questions[question_number], className="research-question"),
                html.P(descriptions[question_number]),
                html.P(f"Error loading visualization: {str(e)}", style={'color': 'red'})
            ])
    # In the get_research_question_layout function, add this for question_number == 2:
    elif question_number == 2:
        try:
            # Load the data
            df, df_filtered = load_and_preprocess_data('data/daily_data_with_lags.csv')
            
            return html.Div([
                html.H2(f"Research Question {question_number}"),
                html.P(questions[question_number], className="research-question"),
                html.P(descriptions[question_number]),
                
                # Add dropdowns for visualization control
                html.Div([
                    html.Div([
                        html.Label("Select Visualization:"),
                        dcc.Dropdown(
                            id='dropdown-viz-q2',
                            options=[
                                {'label': 'Sentiment vs Stock Returns', 'value': 'sentiment_returns'},
                                {'label': 'Sentiment & Stock Price Over Time', 'value': 'time_series'}
                            ],
                            value='sentiment_returns',
                            clearable=False
                        )
                    ], className='dropdown-container', style={'width': '60%', 'display': 'inline-block', 'padding-right': '10px'}),
                    
                    html.Div([
                        html.Label("Select Time Period:"),
                        dcc.Dropdown(
                            id='dropdown-time-q2',
                            options=[
                                {'label': 'Next-Day Returns', 'value': 'next_day'},
                                {'label': 'Same-Day Returns', 'value': 'same_day'}
                            ],
                            value='next_day',
                            clearable=False
                        )
                    ], className='dropdown-container', id='time-period-container', 
                    style={'width': '40%', 'display': 'inline-block'})
                ], style={'display': 'flex'}),
                
                # Visualization container
                html.Div([
                    dcc.Graph(id='graph-q2')
                ], className='viz-container')
            ])
        except Exception as e:
            return html.Div([
                html.H2(f"Research Question {question_number}"),
                html.P(questions[question_number], className="research-question"),
                html.P(descriptions[question_number]),
                html.P(f"Error loading visualization: {str(e)}", style={'color': 'red'})
        ])
    # Modifications to app.py to integrate Research Questions 4 and 5

    # In the get_research_question_layout function, add these blocks for question_number 4 and 5:

    # For Question 4 (Free vs Premium Users):
    elif question_number == 4:
        try:
            return html.Div([
                html.H2(f"Research Question {question_number}"),
                html.P(questions[question_number], className="research-question"),
                html.P(descriptions[question_number]),
                # Keep the existing visualization dropdown
                html.Div([
                    html.Label("Select Visualization:"),
                    dcc.Dropdown(
                        id='dropdown-q4',
                        options=[
                            {'label': 'Revenue Breakdown', 'value': 'revenue'},
                            {'label': 'User Comparison', 'value': 'users'}
                        ],
                        value='revenue',
                        clearable=False
                    )
                ], className='dropdown-container'),
                
                # Keep the visualization container
                html.Div([
                    dcc.Graph(id='graph-q4')
                ], className='viz-container'),
                
                # PDF section
                html.Div([
                    html.H3("Revenue and User Comparison Analysis"),
                    
                    # PDF navigation
                    html.Div([
                        html.P("Quick Access:"),
                        html.Div([
                            html.A("Ad-Supported Revenue", href="#ad-supported-revenue", className="pdf-nav-link"),
                            html.A("Users Comparison", href="#users-comparison", className="pdf-nav-link"),
                            html.A("Premium Revenue", href="#premium-revenue", className="pdf-nav-link"),
                            html.A("Total Revenue", href="#total-revenue", className="pdf-nav-link"),
                        ], className="pdf-navigation")
                    ], className="pdf-nav-container"),
                    
                    # PDF 1: Ad-Supported Revenue
                    html.Div([
                        html.H4("Ad-Supported Revenue Distribution (2019-2024)", id="ad-supported-revenue"),
                        html.Div([
                            html.A(
                                html.Button("Download PDF", className="download-btn"),
                                href="/assets/Ad-Supported-Revenue-Distribution.pdf",
                                download="Ad-Supported-Revenue-Distribution.pdf",
                                target="_blank"
                            ),
                        ], className="pdf-controls"),
                        html.Iframe(
                            src="/assets/Ad-Supported-Revenue-Distribution.pdf",
                            className="pdf-frame"
                        )
                    ], className="pdf-container"),
                    
                    # PDF 2: Users Comparison
                    html.Div([
                        html.H4("Ad-Supported Users vs. Premium Subscribers (2019-2024)", id="users-comparison"),
                        html.Div([
                            html.A(
                                html.Button("Download PDF", className="download-btn"),
                                href="/assets/Ad-Supported-Users-vs-Premium-Subscribers.pdf",
                                download="Ad-Supported-Users-vs-Premium-Subscribers.pdf",
                                target="_blank"
                            ),
                        ], className="pdf-controls"),
                        html.Iframe(
                            src="/assets/Ad-Supported-Users-vs-Premium-Subscribers.pdf",
                            className="pdf-frame"
                        )
                    ], className="pdf-container"),
                    
                    # PDF 3: Premium Revenue
                    html.Div([
                        html.H4("Premium Revenue Distribution (2019-2024)", id="premium-revenue"),
                        html.Div([
                            html.A(
                                html.Button("Download PDF", className="download-btn"),
                                href="/assets/Premium-Revenue-Distribution.pdf",
                                download="Premium-Revenue-Distribution.pdf",
                                target="_blank"
                            ),
                        ], className="pdf-controls"),
                        html.Iframe(
                            src="/assets/Premium-Revenue-Distribution.pdf",
                            className="pdf-frame"
                        )
                    ], className="pdf-container"),
                    
                    # PDF 4: Total Revenue
                    html.Div([
                        html.H4("Spotify Total Revenue Distribution (2019-2024)", id="total-revenue"),
                        html.Div([
                            html.A(
                                html.Button("Download PDF", className="download-btn"),
                                href="/assets/Spotify-Revenue-Distribution.pdf",
                                download="Spotify-Revenue-Distribution.pdf",
                                target="_blank"
                            ),
                        ], className="pdf-controls"),
                        html.Iframe(
                            src="/assets/Spotify-Revenue-Distribution.pdf",
                            className="pdf-frame"
                        )
                    ], className="pdf-container"),
                    
                    # Conclusion
                    html.Div([
                        html.H4("Summary"),
                        html.P([
                            "The analysis clearly demonstrates that while premium subscribers (42% of users) generate approximately 88% of Spotify's revenue, ",
                            "free users still play a vital role in the platform's growth strategy. Both user segments show steady growth from 2019 to 2024, ",
                            "with total revenue reaching €15.67 billion in 2024. The COVID-19 pandemic period (2020-2021) shows notable increases in both users and revenue."
                        ]),
                    ], className="pdf-summary")
                ], className="pdf-section")
            ])
        except Exception as e:
            return html.Div([
                html.H2(f"Research Question {question_number}"),
                html.P(questions[question_number], className="research-question"),
                html.P(descriptions[question_number]),
                html.P(f"Error loading content: {str(e)}", style={'color': 'red'})
            ])
        
    # For Question 5 (Demographics):
    elif question_number == 5:
        try:
            return html.Div([
                html.H2(f"Research Question {question_number}"),
                html.P(questions[question_number], className="research-question"),
                html.P(descriptions[question_number]),
                
                # Add dropdown for visualization selection
                html.Div([
                    html.Label("Select Demographic Factor:"),
                    dcc.Dropdown(
                        id='dropdown-q5',
                        options=[
                            {'label': 'Age Distribution', 'value': 'age'},
                            {'label': 'Gender Distribution', 'value': 'gender'}
                        ],
                        value='age',
                        clearable=False
                    )
                ], className='dropdown-container'),
                
                # Visualization container
                html.Div([
                    dcc.Graph(id='graph-q5')
                ], className='viz-container'),
                
                # PDF section
                html.Div([
                    html.H3("Demographic Analysis"),
                    
                    # PDF navigation
                    html.Div([
                        html.P("Quick Access:"),
                        html.Div([
                            html.A("Age Demographics", href="#age-demographics", className="pdf-nav-link"),
                            html.A("Gender Demographics", href="#gender-demographics", className="pdf-nav-link"),
                        ], className="pdf-navigation")
                    ], className="pdf-nav-container"),
                    
                    # PDF 1: Age Demographics
                    html.Div([
                        html.H4("Spotify's Age Demographics (%)", id="age-demographics"),
                        html.Div([
                            html.A(
                                html.Button("Download PDF", className="download-btn"),
                                href="/assets/Spotify-Age-Demographics.pdf",
                                download="Spotify-Age-Demographics.pdf",
                                target="_blank"
                            ),
                        ], className="pdf-controls"),
                        html.Iframe(
                            src="/assets/Spotify-Age-Demographics.pdf",
                            className="pdf-frame"
                        )
                    ], className="pdf-container"),
                    
                    # PDF 2: Gender Demographics
                    html.Div([
                        html.H4("Spotify's Gender Demographics (%)", id="gender-demographics"),
                        html.Div([
                            html.A(
                                html.Button("Download PDF", className="download-btn"),
                                href="/assets/Spotify-Gender-Demographics.pdf",
                                download="Spotify-Gender-Demographics.pdf",
                                target="_blank"
                            ),
                        ], className="pdf-controls"),
                        html.Iframe(
                            src="/assets/Spotify-Gender-Demographics.pdf",
                            className="pdf-frame"
                        )
                    ], className="pdf-container"),
                    
                    # Summary
                    html.Div([
                        html.H4("Summary"),
                        html.P([
                            "The demographic analysis reveals that Spotify's user base is predominantly young adult to middle-aged, with the 25-34 age group representing the largest segment (29%). ",
                            "Gender distribution shows a slight imbalance, with women accounting for 56% of users compared to 44% for men. ",
                            "These demographic patterns highlight Spotify's strong appeal among younger audiences, while also showing potential growth opportunities among older age groups. ",
                            "The gender distribution may influence content curation strategies and marketing approaches to ensure the platform addresses preferences across all user segments."
                        ]),
                    ], className="pdf-summary")
                ], className="pdf-section")
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

# Update the callback for URL routing to include Questions 3 and 6
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
        return render_question3()  # Call our new function for Question 3
    elif pathname == '/question4':
        return get_research_question_layout(4)
    elif pathname == '/question5':
        return get_research_question_layout(5)
    elif pathname == '/question6':
        return render_question6()  # Call our new function for Question 6
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
        csv_path = 'data/spotify_users.csv'
        df = pd.read_csv(csv_path)
        
        # Create the visualization using the function from q1_visualization.py
        fig = create_user_growth_visualization(df, selected_year)
        save_fig = fig.write_html('graph-q1.html')
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


# Callback to show/hide time period dropdown based on visualization type
@app.callback(
    Output('time-period-container', 'style'),
    [Input('dropdown-viz-q2', 'value')]
)
def toggle_time_period_dropdown(viz_type):
    base_style = {'width': '40%', 'display': 'inline-block'}
    if viz_type == 'sentiment_returns':
        return base_style
    else:
        return {**base_style, 'display': 'none'}

# Callback for research question 2 - News sentiment visualization
@app.callback(
    Output('graph-q2', 'figure'),
    [Input('dropdown-viz-q2', 'value'),
     Input('dropdown-time-q2', 'value')]
)
def update_graph_q2(viz_type, time_period):
    try:
        # Load the data
        df, df_filtered = load_and_preprocess_data('data/daily_data_with_lags.csv')
        
        # Create the visualization
        fig = create_visualization(df_filtered, viz_type, time_period)
        
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

# Callback for research question 4 - Free vs Premium Users visualizations
@app.callback(
    Output('graph-q4', 'figure'),
    [Input('dropdown-q4', 'value')]
)
def update_graph_q4(selected_viz):
    try:
        if selected_viz == 'revenue':
            fig = create_revenue_visualization()
        else:  # users
            fig = create_users_comparison_visualization()
            
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
        
@app.callback(
    [Output('revenue-section', 'style'),
     Output('users-section', 'style')],
    [Input('dropdown-q4', 'value')]
)
def toggle_q4_sections(selected_viz):
    if selected_viz == 'revenue':
        return {'display': 'block'}, {'display': 'none'}
    else:  # users
        return {'display': 'none'}, {'display': 'block'}

@app.callback(
    Output('graph-q5', 'figure'),
    [Input('dropdown-q5', 'value')]
)
def update_graph_q5(selected_demographic):
    try:
        if selected_demographic == 'age':
            fig = create_age_demographics_visualization()
        else:  # gender
            fig = create_gender_demographics_visualization()
            
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

# Add callback for the interactive map in Question 3
@app.callback(
    Output('map-visualization', 'figure'),
    [Input('year-dropdown', 'value')]
)
def update_map_visualization(selected_year):
    return update_map(selected_year)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)