# Swedish Library Data Analysis Dashboard

An interactive dashboard for analyzing Swedish library loan data from 2000 to 2012, featuring a hybrid approach combining direct data analysis and Retrieval Augmented Generation (RAG).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

## Features

### 1. Chatbot Interface
Ask natural language questions about the data and get accurate answers:
- "Which region had the highest number of loans in 2012?"
- "How many total loans were there in Täby for 2008?"
- "What was the trend in library loans over the years?"
- "How did Stockholm's loans change from 2000 to 2012?"

### 2. Data Explorer
Browse and filter the raw data by:
- Region
- Year
- Media Type

View summary statistics and download filtered data as CSV.

### 3. Interactive Dashboards
Visualize the data with interactive charts:
- **Loans by Region**: Compare loan volumes across different regions
- **Loans vs Population**: Analyze the relationship between population size and loan volumes
- **Trends Over Time**: Track changes in loan patterns over the years

## Screenshots

![Chatbot Interface](https://via.placeholder.com/800x400?text=Chatbot+Interface)
![Data Explorer](https://via.placeholder.com/800x400?text=Data+Explorer)
![Interactive Dashboard](https://via.placeholder.com/800x400?text=Interactive+Dashboard)

## Technology Stack

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Sentence Transformers**: Vector embeddings for semantic search
- **ChromaDB**: Vector database for storing embeddings
- **Hugging Face Transformers**: Language models for question answering

## How It Works

The application uses a hybrid approach to answer questions:

1. **Direct Data Analysis**: For specific, structured questions about loans, regions, and years, the system performs direct data analysis using pandas.

2. **Retrieval Augmented Generation (RAG)**: For more contextual questions, the system uses a RAG approach that:
   - Retrieves relevant information from a vector database
   - Generates natural language answers based on the retrieved context

## Data Source

The dataset includes library loan statistics from Swedish municipalities from 2000 to 2012, covering different media types and including population data for per-capita analysis.

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/RezaBaza/swedish-library-dashboard.git
cd swedish-library-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Hugging Face API key:
```
HUGGINGFACE_API_KEY=your_api_key_here
```

4. Run the application:
```bash
streamlit run streamlit_app.py
```

## Deployment

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

[Reza Bazargan](https://github.com/RezaBaza) - Stockholm, Sweden 