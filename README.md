# Swedish Library Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rezabaza-swedish-library-dashboard.streamlit.app/)

An interactive dashboard for exploring Swedish library loan data from 2000-2012, featuring a hybrid question-answering system.

## Features

- **Interactive Chat Interface**: Ask questions about Swedish library data in natural language
- **Data Explorer**: Filter and browse the raw data with customizable filters
- **Interactive Dashboards**: Visualize trends, comparisons, and statistics
- **Hybrid QA System**: Combines direct data analysis with RAG (Retrieval Augmented Generation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RezaBaza/swedish-library-dashboard.git
cd swedish-library-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face API key:
   - Create a `.env` file in the root directory
   - Add your API key: `HUGGINGFACE_API_KEY=your_api_key_here`

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## Deployment

This app can be deployed to Streamlit Cloud. See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## Data Source

The data includes library loan statistics from Swedish municipalities from 2000 to 2012, including:
- Loan counts by region and year
- Population data for per-capita analysis
- Media type breakdowns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Created by [Reza Bazargan](https://github.com/RezaBaza)

## Screenshots

![Chat Interface](screenshots/chat_interface.png)
![Data Explorer](screenshots/data_explorer.png)
![Dashboard](screenshots/dashboard.png)

*Note: Replace the screenshot placeholders with actual screenshots of your application.* 