# California Housing Dashboard 🏡

A comprehensive Streamlit dashboard for exploring the California Housing dataset, combining features from multiple templates.

## Features

- **Introduction Page**: Dataset overview, missing values analysis, and summary statistics
- **Data Exploration**: Full dataset display and detailed statistics
- **Visualization**: Interactive charts including:
  - Bar charts (Seaborn & Streamlit)
  - Line charts
  - Correlation heatmaps
  - Custom visualizations
  - Ocean proximity analysis
- **Automated Report**: Data profiling report (optional, requires additional packages)

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure you have the dataset**:
   - Make sure `housing.csv` is in the same directory as `app.py`

## Running the App

### Local Development

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Deployment Options

#### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy!

#### Other Platforms

- **Heroku**: Use a `Procfile` with `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- **Docker**: Create a Dockerfile with Streamlit
- **AWS/Azure/GCP**: Use container services

## File Structure

```
.
├── app.py              # Main Streamlit application
├── housing.csv         # Dataset file
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Optional Features

### Image Support

If you want to add images to the dashboard:
1. Add `house.png` or `house2.png` to the project directory
2. Uncomment the `st.image()` lines in `app.py` (around lines 30-31)

### Automated Reporting

The automated report feature requires:
- `ydata-profiling`
- `streamlit-pandas-profiling`

These are included in `requirements.txt` but the app will work without them (the feature will be disabled).

## Dataset Information

- **Rows**: 20,640
- **Columns**: 10
- **Features**: 
  - Geographic: longitude, latitude
  - Housing: housing_median_age, total_rooms, total_bedrooms, households
  - Demographics: population, median_income
  - Target: median_house_value
  - Categorical: ocean_proximity

## Troubleshooting

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Dataset Not Found
Ensure `housing.csv` is in the same directory as `app.py`

## License

This project is for educational purposes.
