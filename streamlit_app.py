import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from hybrid_qa import HybridQA
import os

# Initialize session state for chat history and QA system
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "qa_system" not in st.session_state:
    st.session_state.qa_system = HybridQA()
    # Load data
    st.session_state.qa_system.load_data("swedish-library-dashboard/loans_with_population.csv")
    # Store the dataframe for direct access in other pages
    st.session_state.df = st.session_state.qa_system.df

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
                response = st.session_state.qa_system.query(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Show Data page
elif page == "Show Data":
    st.header("Explore the Raw Data")
    
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

# Dashboards page
elif page == "Dashboards":
    st.header("Interactive Dashboards")
    
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
        default=regions[:5]  # Default to first 5 regions
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
        
        # Calculate and display loans per capita
        scatter_data['Loans_per_Capita'] = scatter_data['Loan_Count'] / scatter_data['Population']
        
        # Create bar chart for loans per capita
        fig2 = px.bar(
            scatter_data.sort_values('Loans_per_Capita', ascending=False),
            x='Region_Name',
            y='Loans_per_Capita',
            title=f'Loans per Capita by Region in {selected_year_scatter}',
            labels={'Loans_per_Capita': 'Loans per Capita', 'Region_Name': 'Region'},
            color='Loans_per_Capita',
            color_continuous_scale='Viridis'
        )
        
        # Update layout
        fig2.update_layout(
            xaxis_tickangle=-45,
            height=600
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.subheader("Trends Over Time")
        
        # Create line chart for selected regions over time
        trend_data = region_year_loans.pivot(index='Year', columns='Region_Name', values='Loan_Count').reset_index()
        
        # Create line chart
        fig = go.Figure()
        
        for region in selected_regions:
            if region in trend_data.columns:
                fig.add_trace(go.Scatter(
                    x=trend_data['Year'],
                    y=trend_data[region],
                    mode='lines+markers',
                    name=region
                ))
        
        # Update layout
        fig.update_layout(
            title='Loan Trends Over Time by Region',
            xaxis_title='Year',
            yaxis_title='Total Loans',
            height=600,
            legend_title='Region'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create a normalized trend chart (percentage change from first year)
        st.subheader("Normalized Trends (% Change from First Year)")
        
        # Create normalized line chart
        fig2 = go.Figure()
        
        for region in selected_regions:
            if region in trend_data.columns:
                # Calculate percentage change from first year
                first_year_value = trend_data[region].iloc[0]
                if first_year_value > 0:  # Avoid division by zero
                    normalized_values = ((trend_data[region] / first_year_value) - 1) * 100
                    
                    fig2.add_trace(go.Scatter(
                        x=trend_data['Year'],
                        y=normalized_values,
                        mode='lines+markers',
                        name=region
                    ))
        
        # Update layout
        fig2.update_layout(
            title='Percentage Change in Loans from First Year',
            xaxis_title='Year',
            yaxis_title='% Change from First Year',
            height=600,
            legend_title='Region'
        )
        
        # Add zero line
        fig2.add_shape(
            type="line",
            x0=min(trend_data['Year']),
            y0=0,
            x1=max(trend_data['Year']),
            y1=0,
            line=dict(color="gray", width=1, dash="dash")
        )
        
        st.plotly_chart(fig2, use_container_width=True)

# About section in the sidebar
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("""
This dashboard uses a hybrid approach to analyze Swedish library data:

1. **Direct Data Analysis**: For specific questions about loans, regions, and years.
2. **Retrieval Augmented Generation (RAG)**: For more contextual questions.

The data includes library loan statistics from Swedish municipalities from 2000 to 2012.
""")
