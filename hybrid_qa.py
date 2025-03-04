from transformers import pipeline
import pandas as pd
import os
from library_rag import LibraryRAG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridQA:
    def __init__(self):
        self.rag_system = LibraryRAG()
        
        try:
            # Initialize an open-source language model
            logger.info("Loading language model...")
            self.llm = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_length=512
            )
            logger.info("Language model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading language model: {str(e)}")
            # Fallback to a simpler approach if model loading fails
            self.llm = None
            logger.info("Using fallback approach without language model")
            
        self.df = None

    def load_data(self, csv_path):
        """Load data for both systems"""
        try:
            # Load data for RAG system
            documents = self.rag_system.load_and_process_data(csv_path)
            self.rag_system.add_documents(documents)
            
            # Load data for direct pandas queries
            self.df = pd.read_csv(csv_path)
            logger.info(f"Data loaded successfully. DataFrame shape: {self.df.shape}")
            logger.info(f"Columns: {self.df.columns.tolist()}")
            logger.info(f"Sample regions: {self.df['Region_Name'].unique()[:5]}")
            logger.info(f"Year range: {self.df['Year'].min()} to {self.df['Year'].max()}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            # Create empty DataFrame with expected columns if loading fails
            self.df = pd.DataFrame(columns=['Region_Name', 'Year', 'Loan_Count', 'Media_Type_Desc', 'Population'])
            logger.info("Created empty DataFrame as fallback")

    def query(self, query_text):
        """Process all queries directly with pandas for precise answers"""
        try:
            query_lower = query_text.lower()
            
            # Extract year if present
            years = []
            for word in query_lower.split():
                if word.isdigit() and 1900 <= int(word) <= 2100:
                    years.append(int(word))
            
            # Sort years if multiple found
            years.sort()
                    
            # Check for specific query types
            
            # 1. Region with highest/lowest loans in a year
            if ("region" in query_lower or "municipality" in query_lower) and ("highest" in query_lower or "most" in query_lower) and len(years) > 0:
                year = years[0]  # Use the first year mentioned
                # Filter by year
                year_data = self.df[self.df['Year'] == year]
                if year_data.empty:
                    return f"No data available for the year {year}."
                    
                # Group by region and sum loans
                grouped = year_data.groupby('Region_Name')['Loan_Count'].sum().reset_index()
                # Find the region with the highest loans
                top_region = grouped.loc[grouped['Loan_Count'].idxmax()]
                
                return f"The region with the highest number of loans in {year} was {top_region['Region_Name']} with {int(top_region['Loan_Count']):,} loans."
            
            # 1b. Region with lowest loans in a year
            elif ("region" in query_lower or "municipality" in query_lower) and ("lowest" in query_lower or "least" in query_lower) and len(years) > 0:
                year = years[0]  # Use the first year mentioned
                # Filter by year
                year_data = self.df[self.df['Year'] == year]
                if year_data.empty:
                    return f"No data available for the year {year}."
                    
                # Group by region and sum loans
                grouped = year_data.groupby('Region_Name')['Loan_Count'].sum().reset_index()
                # Find the region with the lowest loans
                bottom_region = grouped.loc[grouped['Loan_Count'].idxmin()]
                
                return f"The region with the lowest number of loans in {year} was {bottom_region['Region_Name']} with {int(bottom_region['Loan_Count']):,} loans."
                
            # 2. Total loans for a specific region and year
            elif "how many" in query_lower and "loans" in query_lower and len(years) > 0:
                year = years[0]  # Use the first year mentioned
                # Try to extract region name
                # This is a simplified approach - in a real app, we'd use NER or fuzzy matching
                for region in self.df['Region_Name'].unique():
                    if region.lower() in query_lower:
                        # Filter data
                        filtered = self.df[(self.df['Year'] == year) & (self.df['Region_Name'] == region)]
                        
                        if filtered.empty:
                            return f"No data found for {region} in {year}."
                        
                        total_loans = filtered['Loan_Count'].sum()
                        return f"There were {int(total_loans):,} total loans in {region} for {year}."
                
                # If we couldn't find a region match
                return "I couldn't identify the region in your question. Please specify a valid Swedish region name."
            
            # 3. Trend analysis
            elif "trend" in query_lower and "loan" in query_lower:
                # Group by year and calculate total loans
                yearly_totals = self.df.groupby('Year')['Loan_Count'].sum().reset_index()
                
                # Determine the trend
                first_year = yearly_totals.iloc[0]
                last_year = yearly_totals.iloc[-1]
                
                if last_year['Loan_Count'] > first_year['Loan_Count']:
                    trend = "increasing"
                else:
                    trend = "decreasing"
                    
                percent_change = ((last_year['Loan_Count'] - first_year['Loan_Count']) / first_year['Loan_Count']) * 100
                
                return f"The overall trend in library loans from {first_year['Year']} to {last_year['Year']} has been {trend}. There was a {abs(percent_change):.1f}% {'increase' if percent_change > 0 else 'decrease'} in total loans over this period."
            
            # 4. Change over time for a specific region
            elif "change" in query_lower and len(years) >= 2:
                # Try to extract region name
                for region in self.df['Region_Name'].unique():
                    if region.lower() in query_lower:
                        # Get data for the specified years
                        start_year = years[0]
                        end_year = years[-1]
                        
                        start_data = self.df[(self.df['Year'] == start_year) & (self.df['Region_Name'] == region)]
                        end_data = self.df[(self.df['Year'] == end_year) & (self.df['Region_Name'] == region)]
                        
                        if start_data.empty or end_data.empty:
                            return f"No data found for {region} in either {start_year} or {end_year}."
                        
                        start_loans = start_data['Loan_Count'].sum()
                        end_loans = end_data['Loan_Count'].sum()
                        
                        change = end_loans - start_loans
                        percent_change = (change / start_loans) * 100 if start_loans > 0 else 0
                        
                        direction = "increased" if change > 0 else "decreased"
                        
                        return f"Loans in {region} {direction} from {int(start_loans):,} in {start_year} to {int(end_loans):,} in {end_year}, a change of {int(abs(change)):,} loans ({abs(percent_change):.1f}% {direction})."
                
                # If we couldn't find a region match
                return "I couldn't identify the region in your question. Please specify a valid Swedish region name."
                
            # Fall back to RAG for other queries
            return self.rag_system.query(query_text)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I'm sorry, I encountered an error while processing your query. Please try a different question or check the data format." 