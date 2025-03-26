
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
