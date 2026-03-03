# How to run

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## What it does

- Pulls Open-Meteo current weather every n seconds
- Stores rows in a DataFrame
- Retrains a LinearRegression model every 10n seconds
- Displays latest data and prediction in Streamlit
