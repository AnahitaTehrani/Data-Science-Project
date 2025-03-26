# app.py - Main application entry point
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

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
        html.H1("Data Science Research Dashboard: Spotify", className='header-title'),
        html.Div([
            dcc.Link('Home', href='/', className='nav-link'),
            dcc.Link('Research Q1', href='/question1', className='nav-link'),
            dcc.Link('Research Q2', href='/question2', className='nav-link'),
            dcc.Link('Research Q3', href='/question3', className='nav-link'),
            dcc.Link('Research Q4', href='/question4', className='nav-link'),
            dcc.Link('Research Q5', href='/question5', className='nav-link'),
            dcc.Link('Research Q6', href='/question6', className='nav-link'),
            dcc.Link('Research Q7', href='/question7', className='nav-link'),
            dcc.Link('Research Q8', href='/question8', className='nav-link'),
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
        html.H2("Welcome to the Data Science Research Dashboard"),
        html.P("This interactive dashboard presents visualizations for eight research questions."),
        html.P("Use the navigation bar above to explore each research question and its corresponding visualizations."),
        
        html.Div([
            html.H3("Research Questions Overview"),
            html.Ul([
                html.Li([html.A("Research Question 1", href="/question1"), 
                         " - Replace with your actual research question title"]),
                html.Li([html.A("Research Question 2", href="/question2"), 
                         " - Replace with your actual research question title"]),
                html.Li([html.A("Research Question 3", href="/question3"), 
                         " - Replace with your actual research question title"]),
                html.Li([html.A("Research Question 4", href="/question4"), 
                         " - Replace with your actual research question title"]),
                html.Li([html.A("Research Question 5", href="/question5"), 
                         " - Replace with your actual research question title"]),
                html.Li([html.A("Research Question 6", href="/question6"), 
                         " - Replace with your actual research question title"]),
                html.Li([html.A("Research Question 7", href="/question7"), 
                         " - Replace with your actual research question title"]),
                html.Li([html.A("Research Question 8", href="/question8"), 
                         " - Replace with your actual research question title"]),
            ])
        ], className='home-overview')
    ])

# Template for a research question page
def get_research_question_layout(question_number):
    return html.Div([
        html.H2(f"Research Question {question_number}"),
        html.P("Replace this with your actual research question text."),
        html.P("Add a brief description or context about this research question here."),
        
        # Example of an interactive visualization with dropdown
        html.Div([
            html.Label("Select Filter:"),
            dcc.Dropdown(
                id=f'dropdown-q{question_number}',
                options=[
                    {'label': '2019', 'value': 2019},
                    {'label': '2020', 'value': 2020},
                    {'label': '2021', 'value': 2021},
                    {'label': 'All Years', 'value': 'all'}
                ],
                value='all',
                clearable=False
            ),
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

# Callback for research question 1 graph
@app.callback(
    Output('graph-q1', 'figure'),
    [Input('dropdown-q1', 'value')]
)
def update_graph_q1(selected_year):
    # Get sample data (to be replaced with actual data)
    df = get_sample_data()
    
    # Filter data based on dropdown selection
    if selected_year != 'all':
        filtered_df = df[df['Year'] == selected_year]
    else:
        filtered_df = df
    
    # Create a bar chart
    fig = px.bar(
        filtered_df, 
        x='Category', 
        y='Values',
        title=f'Sample Bar Chart for Research Question 1 ({selected_year if selected_year != "all" else "All Years"})',
        color='Category'
    )
    
    return fig

# Callback for research question 2 graph
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
    
    # Create a pie chart
    fig = px.pie(
        filtered_df, 
        values='Values', 
        names='Category',
        title=f'Sample Pie Chart for Research Question 2 ({selected_year if selected_year != "all" else "All Years"})'
    )
    
    return fig

# Callback for research question 3 graph
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
    
    # Create a scatter plot
    fig = px.scatter(
        filtered_df, 
        x='Category', 
        y='Values',
        size='Values',
        title=f'Sample Scatter Plot for Research Question 3 ({selected_year if selected_year != "all" else "All Years"})',
        color='Category'
    )
    
    return fig

# Callback for research question 4 graph
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
    
    # Create a line chart
    fig = px.line(
        filtered_df, 
        x='Category', 
        y='Values',
        title=f'Sample Line Chart for Research Question 4 ({selected_year if selected_year != "all" else "All Years"})',
        markers=True
    )
    
    return fig

# Similar callbacks for questions 5-8
# For brevity, we'll define placeholders that you can customize later
for i in range(5, 9):
    @app.callback(
        Output(f'graph-q{i}', 'figure'),
        [Input(f'dropdown-q{i}', 'value')]
    )
    def update_graph(selected_year, question=i):
        # Get sample data (to be replaced with actual data)
        df = get_sample_data()
        
        # Filter data based on dropdown selection
        if selected_year != 'all':
            filtered_df = df[df['Year'] == selected_year]
        else:
            filtered_df = df
        
        # Create different chart types based on question number
        if question == 5:
            # Area chart
            fig = px.area(
                filtered_df, 
                x='Category', 
                y='Values',
                title=f'Sample Area Chart for Research Question {question} ({selected_year if selected_year != "all" else "All Years"})'
            )
        elif question == 6:
            # Histogram
            fig = px.histogram(
                filtered_df, 
                x='Values',
                nbins=5,
                title=f'Sample Histogram for Research Question {question} ({selected_year if selected_year != "all" else "All Years"})'
            )
        elif question == 7:
            # Box plot
            fig = px.box(
                filtered_df, 
                y='Values',
                title=f'Sample Box Plot for Research Question {question} ({selected_year if selected_year != "all" else "All Years"})'
            )
        else:  # question 8
            # Heatmap-like density contour
            fig = px.density_contour(
                filtered_df, 
                x='Category', 
                y='Values',
                title=f'Sample Density Contour for Research Question {question} ({selected_year if selected_year != "all" else "All Years"})'
            )
        
        return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)