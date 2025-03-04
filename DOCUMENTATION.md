# Swedish Library Dashboard: Code Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
   - [Streamlit App](#streamlit-app)
   - [Hybrid Question-Answering System](#hybrid-question-answering-system)
   - [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
4. [Code Walkthrough](#code-walkthrough)
   - [streamlit_app.py](#streamlit_apppy)
   - [hybrid_qa.py](#hybrid_qapy)
   - [library_rag.py](#library_ragpy)
5. [Data Flow](#data-flow)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)
8. [Glossary](#glossary)

## Introduction

The Swedish Library Dashboard is an interactive web application that allows users to explore and analyze Swedish library loan data from 2000-2012. The application combines direct data analysis with a question-answering system powered by Retrieval Augmented Generation (RAG) technology.

Key features include:
- A chat interface for asking natural language questions about the data
- Interactive data exploration with filters and visualizations
- Dashboards showing trends and comparisons across regions and time periods

This documentation explains how the code works, breaking down each component for beginners.

## Project Structure

The project consists of several key files:

```
swedish-library-dashboard/
├── streamlit_app.py         # Main application file
├── hybrid_qa.py             # Hybrid question-answering system
├── library_rag.py           # RAG implementation
├── requirements.txt         # Python dependencies
├── runtime.txt              # Python version specification
├── packages.txt             # System dependencies
├── .streamlit/              # Streamlit configuration
│   ├── config.toml          # App configuration
│   └── secrets.toml         # API keys (template)
├── README.md                # Project overview
└── DEPLOYMENT.md            # Deployment instructions
```

## Core Components

### Streamlit App

[Streamlit](https://streamlit.io/) is a Python framework for creating web applications with minimal code. In this project, Streamlit handles:
- The user interface (UI)
- Interactive elements (buttons, filters, chat input)
- Data visualization
- Page navigation

### Hybrid Question-Answering System

The hybrid QA system combines two approaches:
1. **Direct data analysis**: Uses pandas to answer specific, structured questions by directly querying the data
2. **RAG-based answers**: Uses a language model with retrieved context for more complex or contextual questions

This hybrid approach provides both precise answers for specific queries and flexible, natural language responses for broader questions.

### Retrieval Augmented Generation (RAG)

RAG is a technique that enhances language models by:
1. Retrieving relevant information from a knowledge base
2. Providing this information as context to the language model
3. Generating responses based on both the question and the retrieved context

This approach improves accuracy and reduces hallucinations (made-up information) by grounding the model's responses in actual data.

## Code Walkthrough

### streamlit_app.py

This is the main file that creates the user interface and coordinates the different components.

#### Imports and Setup

```python
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
```

This section imports necessary libraries:
- `streamlit` for the web interface
- `pandas` for data manipulation
- `plotly` for interactive visualizations
- `logging` for error tracking and debugging

#### Error Handling for Imports

```python
try:
    from hybrid_qa import HybridQA
    logger.info("Successfully imported HybridQA")
except Exception as e:
    st.error(f"Error importing HybridQA: {str(e)}")
    logger.error(f"Error importing HybridQA: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Define a fallback class
    class HybridQA:
        # Fallback implementation...
```

This code tries to import the `HybridQA` class. If it fails (which might happen in certain environments), it creates a simplified fallback version to prevent the app from crashing.

#### Session State Initialization

```python
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
            # Error handling for data loading...
```

Streamlit's session state keeps information between reruns of the app. This code:
1. Initializes an empty list for chat messages
2. Creates a QA system if it doesn't exist yet
3. Loads the library data from a CSV file
4. Includes error handling at each step

#### Page Navigation

```python
# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Chatbot", "Show Data", "Dashboards"])
```

This creates a sidebar with radio buttons for navigating between different pages of the app.

#### Chat Interface

```python
# Chatbot page
if page == "Chatbot":
    st.header("Chat Interface")
    
    # Example questions
    st.markdown("### Example questions you can ask:")
    examples = [
        "Which region had the highest number of loans in 2012?",
        # More examples...
    ]
    
    for example in examples:
        st.markdown(f"- *{example}*")
    
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
                    # Error handling...
```

This section creates the chat interface where users can:
1. See example questions for inspiration
2. View the chat history
3. Enter new questions
4. See responses from the QA system

#### Data Explorer

```python
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
        
        # More filter code...
        
        # Filter data based on selections
        filtered_df = st.session_state.df.copy()
        
        if selected_region != "All":
            filtered_df = filtered_df[filtered_df['Region_Name'] == selected_region]
        
        # More filtering code...
        
        # Show data
        st.subheader("Filtered Data")
        st.dataframe(filtered_df)
        
        # Show summary statistics
        # Code for metrics...
        
        # Option to download the filtered data
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f"library_data_{selected_region}_{selected_year}_{selected_media}.csv",
            mime="text/csv",
        )
    except Exception as e:
        # Error handling...
```

This page allows users to:
1. Filter the data by region, year, and media type
2. View the filtered data in a table
3. See summary statistics
4. Download the filtered data as a CSV file

#### Dashboards

```python
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
            default=regions[:5]  # Default to first 5 regions
        )
        
        # Filter data based on selections
        # Filtering code...
        
        # Aggregate data for visualizations
        region_year_loans = filtered_df.groupby(['Region_Name', 'Year'])['Loan_Count'].sum().reset_index()
        
        # Code for different visualization tabs...
```

The dashboards page provides:
1. Multiple tabs for different visualizations
2. Filters for customizing the visualizations
3. Interactive charts created with Plotly

### hybrid_qa.py

This file implements the hybrid question-answering system that combines direct data analysis with RAG.

#### Class Definition and Initialization

```python
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
```

This code:
1. Creates a new RAG system
2. Tries to load a small language model (flan-t5-small)
3. Includes error handling in case the model can't be loaded
4. Initializes `self.df` to store the data

#### Loading Data

```python
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
```

This method:
1. Loads and processes data for the RAG system
2. Loads the same data into a pandas DataFrame for direct queries
3. Logs information about the loaded data
4. Includes error handling to create an empty DataFrame if loading fails

#### Query Processing

```python
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
        
        # More query types...
                
        # Fall back to RAG for other queries
        return self.rag_system.query(query_text)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"I'm sorry, I encountered an error while processing your query. Please try a different question or check the data format."
```

This method:
1. Processes the query text to extract important information (like years)
2. Checks if the query matches specific patterns that can be answered directly
3. For matching patterns, performs direct data analysis using pandas
4. For other queries, falls back to the RAG system
5. Includes error handling to provide a friendly error message if something goes wrong

### library_rag.py

This file implements the Retrieval Augmented Generation (RAG) system for answering more complex or contextual questions.

#### Class Definition and Initialization

```python
class LibraryRAG:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        try:
            # Initialize the embedding model
            logger.info("Importing SentenceTransformer...")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB
            logger.info("Importing and initializing ChromaDB...")
            import chromadb
            from chromadb.utils import embedding_functions
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(
                name="library_data",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name='all-MiniLM-L6-v2'
                )
            )
            self.use_vector_db = True
            logger.info("Vector database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector database: {str(e)}")
            self.use_vector_db = False
            logger.info("Using fallback approach without vector database")
        
        # HuggingFace API endpoint
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        
        # Try to get API key from Streamlit secrets first, then environment variable
        try:
            self.api_key = st.secrets["huggingface"]["api_key"]
            logger.info("Using API key from Streamlit secrets")
        except Exception:
            logger.info("Streamlit secrets not available, trying environment variable")
            self.api_key = os.getenv('HUGGINGFACE_API_KEY')
            
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Response cache
        self.cache = {}
        
        # Store documents for fallback
        self.documents = []
```

This code:
1. Tries to initialize a sentence transformer model for creating embeddings
2. Sets up ChromaDB as a vector database for storing and retrieving documents
3. Configures the Hugging Face API for accessing a larger language model (Mistral-7B)
4. Tries to get the API key from Streamlit secrets or environment variables
5. Initializes a cache for storing responses to avoid repeated API calls
6. Includes error handling and fallback options

#### Loading and Processing Data

```python
def load_and_process_data(self, csv_path):
    """Load and process the CSV data into documents for RAG"""
    logger.info("Reading CSV file...")
    try:
        self.df = pd.read_csv(csv_path)
        documents = []
        
        # Create exact match documents for year-region combinations
        logger.info("Creating exact match documents...")
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
        # More document creation...
        
        # Store documents for fallback
        self.documents = documents
        
        return documents
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        # Create a minimal set of documents for fallback
        self.documents = ["No data available. Please check the data file."]
        return self.documents
```

This method:
1. Loads the CSV data
2. Processes it into different types of documents for the RAG system
3. Creates multiple versions of the same information for better matching
4. Includes error handling to provide a fallback if processing fails

#### Adding Documents to Vector Database

```python
def add_documents(self, documents, batch_size=100):
    """Add documents to the vector store in batches"""
    if not self.use_vector_db:
        logger.info("Vector database not available, skipping document addition")
        return
        
    try:
        logger.info(f"Adding {len(documents)} documents to vector store...")
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.collection.add(
                documents=batch,
                metadatas=[{"source": f"doc_{j}"} for j in range(i, i + len(batch))],
                ids=[f"doc_{j}" for j in range(i, i + len(batch))]
            )
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {str(e)}")
```

This method:
1. Checks if the vector database is available
2. Adds documents to the database in batches to avoid memory issues
3. Includes error handling to log errors if adding documents fails

#### Query Processing

```python
def query(self, query_text, n_results=7):
    """Query the RAG system with caching"""
    # Check cache first
    cache_key = query_text.lower().strip()
    if cache_key in self.cache:
        return self.cache[cache_key]
    
    try:
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
        
        # Get relevant documents
        if self.use_vector_db:
            # Search for relevant documents using vector DB
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Get the relevant documents and prioritize based on query type
            relevant_docs = results['documents'][0]
        else:
            # Fallback to simple keyword matching
            relevant_docs = []
            for doc in self.documents:
                # Simple keyword matching
                if any(keyword in doc.lower() for keyword in query_lower.split()):
                    relevant_docs.append(doc)
            
            # Limit to n_results
            relevant_docs = relevant_docs[:n_results]
        
        # Prioritize documents based on query type
        # Code for prioritizing documents...
        
        # Use more context for better answers
        context = "\n".join(relevant_docs[:5]) if relevant_docs else "No relevant information found."
        
        # Create prompt for the LLM
        prompt = f"""<s>[INST] You are a helpful assistant that answers questions about Swedish library data. 
        Answer the question using ONLY the following context. If you cannot find the exact answer, say so clearly.
        Be concise but complete - provide the full answer with specific numbers when available.

Context: {context}

Question: {query_text}

Answer: [/INST]</s>"""
        
        # Generate response using HuggingFace API
        try:
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
                },
                timeout=10  # Add timeout to prevent hanging
            )
            
            if response.status_code == 200:
                response_text = response.json()[0]['generated_text'].strip()
                # Cache the response
                self.cache[cache_key] = response_text
                return response_text
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                # Fallback response
                fallback = self.generate_fallback_response(query_text, context)
                self.cache[cache_key] = fallback
                return fallback
        except Exception as e:
            logger.error(f"Error calling HuggingFace API: {str(e)}")
            # Fallback response
            fallback = self.generate_fallback_response(query_text, context)
            self.cache[cache_key] = fallback
            return fallback
    except Exception as e:
        logger.error(f"Error in query processing: {str(e)}")
        return "I'm sorry, I encountered an error while processing your query. Please try a different question."
```

This method:
1. Checks if the query is already in the cache to avoid repeated processing
2. Determines the query type to prioritize relevant documents
3. Retrieves relevant documents from the vector database or falls back to keyword matching
4. Creates a prompt for the language model with the retrieved context
5. Calls the Hugging Face API to generate a response
6. Includes error handling and fallback options at multiple levels

#### Fallback Response Generation

```python
def generate_fallback_response(self, query, context):
    """Generate a simple response when the API fails"""
    # Extract the most relevant document from context
    lines = context.split('\n')
    if not lines or lines[0] == "No relevant information found.":
        return "I'm sorry, I don't have enough information to answer that question accurately."
    
    # Find the most relevant line based on simple keyword matching
    query_words = set(query.lower().split())
    best_line = None
    best_score = 0
    
    for line in lines:
        line_words = set(line.lower().split())
        common_words = query_words.intersection(line_words)
        score = len(common_words)
        
        if score > best_score:
            best_score = score
            best_line = line
    
    if best_line:
        # Clean up the line (remove prefixes like EXACT:, RANK:, etc.)
        for prefix in ["EXACT:", "RANK:", "TREND:", "OVERALL:", "DETAIL:"]:
            if best_line.startswith(prefix):
                best_line = best_line[len(prefix):].strip()
        
        return f"Based on the available data: {best_line}"
    else:
        return "I'm sorry, I don't have enough information to answer that question accurately."
```

This method:
1. Provides a fallback when the API call fails
2. Uses simple keyword matching to find the most relevant document
3. Cleans up the document to create a readable response
4. Returns a friendly message if no relevant information is found

## Data Flow

Here's how data flows through the application:

1. **Data Loading**:
   - CSV data is loaded into a pandas DataFrame
   - The same data is processed into documents for the RAG system
   - Documents are added to the vector database (ChromaDB)

2. **User Query**:
   - User enters a question in the chat interface
   - The question is sent to the hybrid QA system

3. **Query Processing**:
   - The system checks if the query matches specific patterns
   - For matching patterns, it performs direct data analysis
   - For other queries, it uses the RAG system:
     - Retrieves relevant documents from the vector database
     - Sends the query and context to the language model
     - Returns the generated response

4. **Response Display**:
   - The response is displayed in the chat interface
   - The chat history is updated

## Deployment

The application is deployed on Streamlit Cloud, which provides:
- Hosting for the web application
- Automatic updates when the GitHub repository changes
- Secret management for API keys

For detailed deployment instructions, see the [DEPLOYMENT.md](DEPLOYMENT.md) file.

## Troubleshooting

Common issues and solutions:

1. **PyTorch Compatibility Issues**:
   - Use Python 3.9 instead of newer versions
   - Specify exact versions of PyTorch in requirements.txt

2. **API Key Issues**:
   - Ensure the Hugging Face API key is correctly set in Streamlit secrets
   - Check that the API key has access to the required models

3. **Memory Errors**:
   - Increase the memory allocation in Streamlit Cloud settings
   - Process data in smaller batches

4. **Missing Data**:
   - Check that the CSV file is correctly uploaded and accessible
   - Verify the file path in the code

## Glossary

- **Streamlit**: A Python framework for creating web applications
- **Pandas**: A library for data manipulation and analysis
- **RAG (Retrieval Augmented Generation)**: A technique that enhances language models by providing relevant context
- **Vector Database**: A database that stores and retrieves data based on semantic similarity
- **Embeddings**: Numerical representations of text that capture semantic meaning
- **Language Model**: An AI model that can generate and understand text
- **API (Application Programming Interface)**: A way for different software systems to communicate
- **Session State**: Streamlit's way of preserving data between reruns of the app 