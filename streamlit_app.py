import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page configuration - This should be the first Streamlit command
st.set_page_config(
    page_title="Swedish Library Analysis",
    page_icon="📚",
    layout="wide"  # Uses the full screen width
)

# Add title and description
st.title("📚 Swedish Library Loans Analysis")
st.markdown("""
This dashboard analyzes library loans across different regions in Sweden.
Use the filters below to explore the data and discover trends.
""")

# Function to load data - In Streamlit, it's good practice to cache data loading
@st.cache_data  # This decorator caches the data to avoid reloading on every interaction
def load_data():
    return pd.read_csv("loans_with_population.csv")  # Using relative path for cloud deployment

# Load the data
df = load_data()

# Helper functions
def calculate_yoy_changes(df, region):
    """Calculate year-over-year changes for a specific region"""
    region_data = df[df['Region_Name'] == region].copy()
    yearly_loans = region_data.groupby('Year')['Loan_Count'].sum().reset_index()
    yearly_population = region_data.groupby('Year')['Population'].first().reset_index()
    
    yearly_loans['YoY_Loans'] = yearly_loans['Loan_Count'].pct_change() * 100
    yearly_population['YoY_Population'] = yearly_population['Population'].pct_change() * 100
    
    return yearly_loans.merge(yearly_population, on='Year')

# Create sidebar for filters
st.sidebar.header("Filters")

# Region selection
regions = sorted(df['Region_Name'].unique())
selected_region = st.sidebar.selectbox(
    "Select Region",
    ['All Regions'] + regions,
    index=0
)

# Media type selection
media_types = sorted(df['Media_Type_Desc'].unique())
selected_media = st.sidebar.selectbox(
    "Select Media Type",
    ['All Media Types'] + media_types,
    index=0
)

# Year selection
years = sorted(df['Year'].unique())
selected_year = st.sidebar.selectbox(
    "Select Year",
    years,
    index=len(years)-1  # Default to latest year
)

# Apply filters
filtered_df = df.copy()
if selected_region != 'All Regions':
    filtered_df = filtered_df[filtered_df['Region_Name'] == selected_region]
if selected_media != 'All Media Types':
    filtered_df = filtered_df[filtered_df['Media_Type_Desc'] == selected_media]

# Create two columns for metrics
col1, col2 = st.columns(2)

# Display key metrics
with col1:
    # Get unique population per region for the selected year (avoid double counting)
    total_population = (filtered_df[filtered_df['Year'] == selected_year]
                       .groupby('Region_Name')['Population']
                       .first()  # Take just one population value per region
                       .sum())
    st.metric(
        label="Total Population",
        value=f"{total_population:,.0f}"
    )

with col2:
    total_loans = filtered_df[filtered_df['Year'] == selected_year]['Loan_Count'].sum()
    st.metric(
        label="Total Loans",
        value=f"{total_loans:,.0f}"
    )

# Create YoY comparison graph
st.subheader("Year-over-Year Changes")
st.markdown("This graph shows how population and loans change over time.")

if selected_region != 'All Regions':
    yearly_data = calculate_yoy_changes(filtered_df, selected_region)
else:
    # Calculate for all regions combined
    yearly_data = filtered_df.groupby('Year').agg({
        'Loan_Count': 'sum',
        'Population': 'sum'
    }).reset_index()
    yearly_data['YoY_Loans'] = yearly_data['Loan_Count'].pct_change() * 100
    yearly_data['YoY_Population'] = yearly_data['Population'].pct_change() * 100

# Create the dual-axis figure
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add Population line
fig.add_trace(
    go.Scatter(
        x=yearly_data['Year'],
        y=yearly_data['YoY_Population'],
        name='Population Change %',
        line=dict(color='blue', width=2),
        hovertemplate='Year: %{x}<br>Population Change: %{y:.1f}%<extra></extra>'
    ),
    secondary_y=False
)

# Add Loans line
fig.add_trace(
    go.Scatter(
        x=yearly_data['Year'],
        y=yearly_data['YoY_Loans'],
        name='Loans Change %',
        line=dict(color='red', width=2),
        hovertemplate='Year: %{x}<br>Loans Change: %{y:.1f}%<extra></extra>'
    ),
    secondary_y=True
)

# Update layout
fig.update_layout(
    title=f'Year-over-Year Changes - {selected_region}' +
          (f' ({selected_media})' if selected_media != 'All Media Types' else ''),
    hovermode='x unified',
    showlegend=True,
    plot_bgcolor='white',
    paper_bgcolor='white'
)

# Update axes
fig.update_xaxes(title_text="Year", showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.update_yaxes(
    title_text="Population Change (%)",
    showgrid=True,
    gridwidth=1,
    gridcolor='lightgray',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='gray',
    secondary_y=False
)
fig.update_yaxes(
    title_text="Loans Change (%)",
    showgrid=True,
    gridwidth=1,
    gridcolor='lightgray',
    zeroline=True,
    zerolinewidth=1,
    zerolinecolor='gray',
    secondary_y=True
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Add regional comparison
st.subheader("Regional Comparison")
if selected_region == 'All Regions':
    # Calculate loans per capita for each region
    region_stats = filtered_df[filtered_df['Year'] == selected_year].groupby('Region_Name').agg({
        'Loan_Count': 'sum',
        'Population': 'first'
    }).reset_index()
    
    region_stats['Loans_per_Capita'] = region_stats['Loan_Count'] / region_stats['Population']
    
    # Create bar chart
    fig_regions = go.Figure()
    fig_regions.add_trace(
        go.Bar(
            y=region_stats['Region_Name'],
            x=region_stats['Loans_per_Capita'],
            orientation='h',
            text=region_stats['Loans_per_Capita'].round(2),
            textposition='auto',
        )
    )
    
    fig_regions.update_layout(
        title='Loans per Capita by Region',
        xaxis_title='Loans per Capita',
        yaxis_title='Region',
        height=max(600, len(region_stats) * 20),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(fig_regions, use_container_width=True)

# Add footer with data source information
st.markdown("---")
st.markdown("""
**Data Source**: Swedish Library Statistics  
**Last Updated**: 2024
""") 