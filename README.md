# Swedish Library Analysis Dashboard

An interactive dashboard analyzing library loans across different regions in Sweden. Built with Streamlit and Plotly.

## Live Demo
🔗 [View the Live Dashboard](https://swedish-library-dashboard-rezabaza.streamlit.app/)

## Features
- Year-over-Year comparison of population and library loans
- Regional comparison of loans per capita
- Interactive filters for regions, media types, and years
- Dynamic visualizations with dual-axis graphs
- Real-time metric calculations
- Responsive design for all devices

## Data Analysis
The dashboard provides insights into:
- Population trends across Swedish regions
- Library loan patterns over time
- Media type preferences by region
- Per capita library usage
- Year-over-Year (YoY) change comparisons

## Data Source
Swedish Library Statistics (2000-2024)
- Population data by region
- Library loans by media type
- Regional statistics

## Technical Details
### Built With
- Python 3.7+
- Streamlit for web interface
- Plotly for interactive visualizations
- Pandas for data manipulation
- NumPy for numerical operations

### Local Setup
1. Clone this repository:
```bash
git clone https://github.com/RezaBaza/swedish-library-dashboard.git
cd swedish-library-dashboard
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run streamlit_app.py
```

## Usage
1. Select a region from the dropdown to focus on specific areas
2. Choose a media type to analyze particular loan categories
3. Use the year selector to view data for specific time periods
4. Hover over graphs to see detailed information
5. Compare different regions' performance using the per capita analysis

## Updates and Maintenance
The dashboard is automatically updated when changes are pushed to the main branch of this repository.
