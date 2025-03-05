import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize with error handling
try:
    from hybrid_qa import HybridQA
    logger.info("Successfully imported HybridQA")
except Exception as e:
    st.error(f"Error importing HybridQA: {str(e)}")
    logger.error(f"Error importing HybridQA: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Define a fallback class
    class HybridQA:
        def __init__(self):
            self.df = None
            
        def load_data(self, csv_path):
            try:
                self.df = pd.read_csv(csv_path)
                logger.info(f"Loaded data with shape: {self.df.shape}")
            except Exception as e:
                logger.error(f"Error loading data: {str(e)}")
                # Create sample data
                self.df = pd.DataFrame({
                    'Region_Name': ['Sample Region'],
                    'Year': [2020],
                    'Loan_Count': [1000],
                    'Media_Type_Desc': ['Books'],
                    'Population': [10000]
                })
                
        def query(self, query_text):
            return "I'm sorry, the QA system is currently unavailable. Please try exploring the data directly using the 'Show Data' tab."

# Set page config
st.set_page_config(
    page_title="Municipality Library Data Analysis, Sweden",
    page_icon="📚",
    layout="wide"
)

# Initialize session state for chat history and QA system
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "qa_system" not in st.session_state:
    try:
        logger.info("Initializing QA system")
        st.session_state.qa_system = HybridQA()
        
        # Load data with error handling
        try:
            data_path = "swedish-library-dashboard/loans_with_population.csv"
            if not os.path.exists(data_path):
                # Try alternative path
                data_path = "loans_with_population.csv"
                
            logger.info(f"Loading data from: {data_path}")
            st.session_state.qa_system.load_data(data_path)
            # Store the dataframe for direct access in other pages
            st.session_state.df = st.session_state.qa_system.df
            logger.info("Data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            st.error(f"Error loading data: {str(e)}")
            # Create sample data
            st.session_state.df = pd.DataFrame({
                'Region_Name': ['Sample Region'],
                'Year': [2020],
                'Loan_Count': [1000],
                'Media_Type_Desc': ['Books'],
                'Population': [10000]
            })
    except Exception as e:
        logger.error(f"Error initializing QA system: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error initializing QA system: {str(e)}")

# Set the title of the app
st.title("Swedish Library Data Analysis")

# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Chatbot", "Show Data", "Dashboards"])

# Chatbot page
if page == "Chatbot":
    st.header("Chat Interface")
    
    # Example questions
    st.markdown("### Example questions you can ask:")
    examples = [
        "Which region had the highest number of loans in 2012?",
        "How many total loans were there in Täby for 2008?",
        "What was the trend in library loans over the years?",
        "Which region had the lowest number of loans in 2010?",
        "How did Stockholm's loans change from 2000 to 2012?"
    ]
    
    for example in examples:
        st.markdown(f"- *{example}*")
    
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about Swedish library data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_system.query(prompt)
                    st.markdown(response)
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    st.error(error_msg)
                    response = "I'm sorry, I encountered an error while processing your question. Please try a different question or check the data directly in the 'Show Data' tab."
                    st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Show Data page
elif page == "Show Data":
    st.header("Explore the Raw Data")
    
    try:
        # Get unique values for filters
        regions = sorted(st.session_state.df['Region_Name'].unique())
        years = sorted(st.session_state.df['Year'].unique())
        media_types = sorted(st.session_state.df['Media_Type_Desc'].unique())
        
        # Create filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_region = st.selectbox("Select Region", ["All"] + regions)
        
        with col2:
            selected_year = st.selectbox("Select Year", ["All"] + years)
        
        with col3:
            selected_media = st.selectbox("Select Media Type", ["All"] + media_types)
        
        # Filter data based on selections
        filtered_df = st.session_state.df.copy()
        
        if selected_region != "All":
            filtered_df = filtered_df[filtered_df['Region_Name'] == selected_region]
        
        if selected_year != "All":
            filtered_df = filtered_df[filtered_df['Year'] == selected_year]
        
        if selected_media != "All":
            filtered_df = filtered_df[filtered_df['Media_Type_Desc'] == selected_media]
        
        # Show data
        st.subheader("Filtered Data")
        st.dataframe(filtered_df)
        
        # Show summary statistics
        st.subheader("Summary Statistics")
        
        # Calculate total loans
        total_loans = filtered_df['Loan_Count'].sum()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Loans", f"{total_loans:,.0f}")
        
        with col2:
            if 'Population' in filtered_df.columns and filtered_df['Population'].sum() > 0:
                avg_loans_per_capita = total_loans / filtered_df['Population'].sum()
                st.metric("Avg. Loans per Capita", f"{avg_loans_per_capita:.2f}")
        
        with col3:
            if selected_region != "All" and selected_year != "All":
                # Calculate percentage of total for that year
                year_total = st.session_state.df[st.session_state.df['Year'] == selected_year]['Loan_Count'].sum()
                percentage = (total_loans / year_total) * 100 if year_total > 0 else 0
                st.metric("% of Year Total", f"{percentage:.2f}%")
        
        # Option to download the filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f"library_data_{selected_region}_{selected_year}_{selected_media}.csv",
            mime="text/csv",
        )
    except Exception as e:
        logger.error(f"Error in Show Data page: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error displaying data: {str(e)}")

# Dashboards page
elif page == "Dashboards":
    st.header("Interactive Dashboards")
    
    try:
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Loans by Region", "Loans vs Population", "Trends Over Time"])
        
        # Get unique values for filters
        regions = sorted(st.session_state.df['Region_Name'].unique())
        years = sorted(st.session_state.df['Year'].unique())
        
        # Sidebar filters for dashboards
        st.sidebar.header("Dashboard Filters")
        
        # Year range slider
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years)))
        )
        
        # Multi-select for regions
        selected_regions = st.sidebar.multiselect(
            "Select Regions",
            regions,
            default=["Stockholm", "Malmö", "Göteborg", "Lund", "Uppsala", "Örebro", "Linköping"]  # Specific default regions
        )
        
        # Filter data based on selections
        filtered_df = st.session_state.df[
            (st.session_state.df['Year'] >= year_range[0]) & 
            (st.session_state.df['Year'] <= year_range[1])
        ]
        
        if selected_regions:
            filtered_df = filtered_df[filtered_df['Region_Name'].isin(selected_regions)]
        
        # Aggregate data for visualizations
        region_year_loans = filtered_df.groupby(['Region_Name', 'Year'])['Loan_Count'].sum().reset_index()
        
        with tab1:
            st.subheader("Loans by Region")
            
            # Select specific year for the bar chart
            selected_year_bar = st.slider(
                "Select Year for Bar Chart",
                min_value=year_range[0],
                max_value=year_range[1],
                value=year_range[1]
            )
            
            # Filter for selected year
            year_data = region_year_loans[region_year_loans['Year'] == selected_year_bar]
            
            # Sort by loan count
            year_data = year_data.sort_values('Loan_Count', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                year_data,
                x='Region_Name',
                y='Loan_Count',
                title=f'Total Loans by Region in {selected_year_bar}',
                labels={'Loan_Count': 'Total Loans', 'Region_Name': 'Region'},
                color='Loan_Count',
                color_continuous_scale='Viridis'
            )
            
            # Update layout
            fig.update_layout(
                xaxis_tickangle=-45,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Loans vs Population")
            
            # Select specific year for the scatter plot
            selected_year_scatter = st.slider(
                "Select Year for Scatter Plot",
                min_value=year_range[0],
                max_value=year_range[1],
                value=year_range[1]
            )
            
            # Prepare data for scatter plot
            scatter_data = filtered_df[filtered_df['Year'] == selected_year_scatter]
            
            # Aggregate by region
            scatter_data = scatter_data.groupby('Region_Name').agg({
                'Loan_Count': 'sum',
                'Population': 'mean'  # Using mean as population should be the same for all entries of a region in a year
            }).reset_index()
            
            # Create scatter plot
            fig = px.scatter(
                scatter_data,
                x='Population',
                y='Loan_Count',
                title=f'Loans vs Population in {selected_year_scatter}',
                labels={'Loan_Count': 'Total Loans', 'Population': 'Population'},
                color='Loan_Count',
                size='Population',
                hover_name='Region_Name',
                color_continuous_scale='Viridis'
            )
            
            # Add trendline
            fig.update_layout(height=600)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Trends Over Time")
            
            # Aggregate data by year
            yearly_totals = filtered_df.groupby('Year')['Loan_Count'].sum().reset_index()
            
            # Create line chart
            fig = px.line(
                yearly_totals,
                x='Year',
                y='Loan_Count',
                title='Total Loans Over Time',
                labels={'Loan_Count': 'Total Loans', 'Year': 'Year'},
                markers=True
            )
            
            # Update layout
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Region comparison
            if selected_regions:
                st.subheader("Region Comparison")
                
                # Prepare data for region comparison
                region_trends = filtered_df.groupby(['Region_Name', 'Year'])['Loan_Count'].sum().reset_index()
                
                # Create line chart for region comparison
                fig = px.line(
                    region_trends,
                    x='Year',
                    y='Loan_Count',
                    color='Region_Name',
                    title='Loans by Region Over Time',
                    labels={'Loan_Count': 'Total Loans', 'Year': 'Year', 'Region_Name': 'Region'},
                    markers=True
                )
                
                # Update layout
                fig.update_layout(height=600)
                
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Error in Dashboards page: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error displaying dashboards: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("### About this app")
st.markdown("""
This dashboard provides analysis of Swedish library loan data from 2000-2012. 
It combines direct data analysis with a question-answering system to help you explore the data.
Data source: [SCB](https://www.statistikdatabasen.scb.se/pxweb/sv/ssd/).
""")

# Add version info
st.sidebar.markdown("---")
st.sidebar.info("Version 1.0.0 | Created by Reza Bazargan")
