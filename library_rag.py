import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import requests
import json
import os
from dotenv import load_dotenv
from tqdm import tqdm

class LibraryRAG:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="library_data",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='all-MiniLM-L6-v2'
            )
        )
        
        # HuggingFace API endpoint
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        self.headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
        
        # Response cache
        self.cache = {}

    def load_and_process_data(self, csv_path):
        """Load and process the CSV data into documents for RAG"""
        print("Reading CSV file...")
        self.df = pd.read_csv(csv_path)
        documents = []
        
        # Create exact match documents for year-region combinations
        print("Creating exact match documents...")
        for region in self.df['Region_Name'].unique():
            region_data = self.df[self.df['Region_Name'] == region]
            for year in sorted(region_data['Year'].unique()):
                year_data = region_data[region_data['Year'] == year]
                total_loans = year_data['Loan_Count'].sum()
                
                # Create three versions of the same information for better matching
                docs = [
                    f"EXACT: The total loans in {region} for {year} was {total_loans:,}.",
                    f"EXACT: In {year}, {region} had {total_loans:,} total loans.",
                    f"EXACT: {region} {year} total loans: {total_loans:,}"
                ]
                documents.extend(docs)
                
                # Add a detailed breakdown document
                breakdown = f"DETAIL: {region} {year} breakdown - "
                for _, row in year_data.iterrows():
                    breakdown += f"{row['Media_Type_Desc']}: {row['Loan_Count']:,}, "
                documents.append(breakdown.strip(", "))
        
        # Create highest/lowest documents for each year
        print("Creating ranking documents...")
        for year in sorted(self.df['Year'].unique()):
            year_data = self.df[self.df['Year'] == year]
            by_region = year_data.groupby('Region_Name')['Loan_Count'].sum().reset_index()
            
            # Highest loans
            top_region = by_region.loc[by_region['Loan_Count'].idxmax()]
            highest_doc = f"RANK: The region with the highest number of loans in {year} was {top_region['Region_Name']} with {top_region['Loan_Count']:,} loans."
            documents.append(highest_doc)
            
            # Lowest loans
            bottom_region = by_region.loc[by_region['Loan_Count'].idxmin()]
            lowest_doc = f"RANK: The region with the lowest number of loans in {year} was {bottom_region['Region_Name']} with {bottom_region['Loan_Count']:,} loans."
            documents.append(lowest_doc)
        
        # Create trend documents (one per region)
        print("Creating trend documents...")
        for region in self.df['Region_Name'].unique():
            region_data = self.df[self.df['Region_Name'] == region]
            trend = f"TREND: {region} loans by year - "
            for year in sorted(region_data['Year'].unique()):
                total = region_data[region_data['Year'] == year]['Loan_Count'].sum()
                trend += f"{year}: {total:,}, "
            documents.append(trend.strip(", "))
        
        # Create overall trend document
        yearly_totals = self.df.groupby('Year')['Loan_Count'].sum().reset_index()
        first_year = yearly_totals.iloc[0]
        last_year = yearly_totals.iloc[-1]
        percent_change = ((last_year['Loan_Count'] - first_year['Loan_Count']) / first_year['Loan_Count']) * 100
        trend_direction = "increasing" if percent_change > 0 else "decreasing"
        
        overall_trend = f"OVERALL: The overall trend in library loans from {first_year['Year']} to {last_year['Year']} has been {trend_direction}. "
        overall_trend += f"There was a {abs(percent_change):.1f}% {'increase' if percent_change > 0 else 'decrease'} in total loans over this period."
        documents.append(overall_trend)
        
        return documents

    def add_documents(self, documents, batch_size=100):
        """Add documents to the vector store in batches"""
        print(f"Adding {len(documents)} documents to vector store...")
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i + batch_size]
            self.collection.add(
                documents=batch,
                metadatas=[{"source": f"doc_{j}"} for j in range(i, i + len(batch))],
                ids=[f"doc_{j}" for j in range(i, i + len(batch))]
            )

    def query(self, query_text, n_results=7):
        """Query the RAG system with caching"""
        # Check cache first
        cache_key = query_text.lower().strip()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Extract year and region from query if present
        query_lower = query_text.lower()
        
        # Determine query type for better document retrieval
        query_type = "general"
        if "highest" in query_lower or "most" in query_lower or "top" in query_lower:
            query_type = "highest"
        elif "lowest" in query_lower or "least" in query_lower or "bottom" in query_lower:
            query_type = "lowest"
        elif "trend" in query_lower or "change" in query_lower or "over the years" in query_lower:
            query_type = "trend"
        
        # Search for relevant documents
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        # Get the relevant documents and prioritize based on query type
        relevant_docs = results['documents'][0]
        
        # Prioritize documents based on query type
        if query_type == "highest" or query_type == "lowest":
            # For highest/lowest queries, prioritize RANK documents
            relevant_docs = sorted(relevant_docs, 
                                key=lambda x: (x.startswith("RANK"), "highest" in x.lower() if query_type == "highest" else "lowest" in x.lower()),
                                reverse=True)
        elif query_type == "trend":
            # For trend queries, prioritize TREND and OVERALL documents
            relevant_docs = sorted(relevant_docs, 
                                key=lambda x: (x.startswith("TREND") or x.startswith("OVERALL")),
                                reverse=True)
        else:
            # For other queries, prioritize EXACT matches
            relevant_docs = sorted(relevant_docs, 
                                key=lambda x: x.startswith("EXACT"), 
                                reverse=True)
        
        # Use more context for better answers
        context = "\n".join(relevant_docs[:5])
        
        # Create prompt for the LLM
        prompt = f"""<s>[INST] You are a helpful assistant that answers questions about Swedish library data. 
        Answer the question using ONLY the following context. If you cannot find the exact answer, say so clearly.
        Be concise but complete - provide the full answer with specific numbers when available.

Context: {context}

Question: {query_text}

Answer: [/INST]</s>"""
        
        # Generate response using HuggingFace API
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "return_full_text": False
                }
            }
        )
        
        if response.status_code == 200:
            response_text = response.json()[0]['generated_text'].strip()
            # Cache the response
            self.cache[cache_key] = response_text
            return response_text
        else:
            return f"Error: {response.status_code} - {response.text}"

    def create_documents(self):
        """Create documents from the dataframe"""
        documents = []
        
        # Create overall trend documents
        for region in self.df['Region_Name'].unique():
            region_data = self.df[self.df['Region_Name'] == region]
            for year in region_data['Year'].unique():
                year_data = region_data[region_data['Year'] == year]
                total_loans = year_data['Loan_Count'].sum()
                
                # Create detailed yearly summary
                doc = f"In {year}, {region} had exactly {total_loans:,} total loans. "
                doc += "The breakdown by media type was: "
                
                # Add media type breakdown
                for _, row in year_data.iterrows():
                    doc += f"{row['Media_Type_Desc']}: {row['Loan_Count']:,} loans, "
                
                # Add per capita information if available
                if 'Population' in year_data.columns and year_data['Population'].iloc[0] > 0:
                    population = year_data['Population'].iloc[0]
                    loans_per_capita = total_loans / population
                    doc += f"With a population of {population:,}, this represents {loans_per_capita:.2f} loans per capita."
                
                documents.append(doc.strip())
        
        # Create media type trend documents
        for media_type in self.df['Media_Type_Desc'].unique():
            media_data = self.df[self.df['Media_Type_Desc'] == media_type]
            doc = f"Trend for {media_type}: "
            
            for year in sorted(media_data['Year'].unique()):
                year_total = media_data[media_data['Year'] == year]['Loan_Count'].sum()
                doc += f"In {year}: {year_total:,} loans, "
            
            documents.append(doc.strip())
        
        return documents 