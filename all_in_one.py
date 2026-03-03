"""Single-file version of the project including instructions.

How to run

    pip install -r requirements.txt
    streamlit run all_in_one.py --server.address 0.0.0.0 --server.port 8501

What it does

- Pulls Open-Meteo current weather every n seconds
- Stores rows in a DataFrame
- Retrains a LinearRegression model every 10n seconds
- Displays latest data and prediction in Streamlit

Dependencies (requirements.txt):
    streamlit
    pandas
    requests
    scikit-learn
    joblib

"""

import time
import requests
import pandas as pd
import joblib
import streamlit as st
from sklearn.linear_model import LinearRegression

# backend functionality
LAT, LON = 41.8781, -87.6298
TZ = "America/Chicago"
_last_good = None

def fetch_row():
    global _last_good

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,pressure_msl"
        f"&timezone={TZ}"
    )

    last_err = None
    for _ in range(3):
        try:
            r = requests.get(url, timeout=45)
            r.raise_for_status()
            cur = r.json()["current"]

            row = {
                "time": cur["time"],
                "temperature_2m": float(cur["temperature_2m"]),
                "relative_humidity_2m": float(cur["relative_humidity_2m"]),
                "precipitation": float(cur["precipitation"]),
                "wind_speed_10m": float(cur["wind_speed_10m"]),
                "pressure_msl": float(cur["pressure_msl"]),
                "fetched_at": time.time()
            }
            _last_good = row
            return row
        except Exception as e:
            last_err = e
            time.sleep(1)

    if _last_good is not None:
        fallback = dict(_last_good)
        fallback["fetched_at"] = time.time()
        fallback["error"] = str(last_err)
        return fallback

    raise last_err


def update_model(df, model_path="model.joblib"):
    X = df[["relative_humidity_2m", "precipitation", "wind_speed_10m", "pressure_msl"]]
    y = df["temperature_2m"]
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model

# streamlit app
st.set_page_config(page_title="Real-Time Weather Predictor", layout="wide")

n = st.sidebar.number_input("Fetch interval (seconds)", min_value=1, max_value=60, value=5)
model_every = 10 * n

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "last_train" not in st.session_state:
    st.session_state.last_train = 0.0
if "model" not in st.session_state:
    st.session_state.model = None

st.title("Real-Time Data + Prediction Dashboard")
st.write("API: Open-Meteo current weather. Target: temperature_2m.")

pred_box = st.empty()
df_box = st.empty()

while True:
    row = fetch_row()
    st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([row])], ignore_index=True)

    if time.time() - st.session_state.last_train >= model_every and len(st.session_state.df) >= 5:
        st.session_state.model = update_model(st.session_state.df)
        st.session_state.last_train = time.time()

    if st.session_state.model is None:
        pred_box.metric("Prediction", "Model not trained yet.")
    else:
        latest = st.session_state.df.iloc[-1]
        X_latest = [[latest["relative_humidity_2m"], latest["precipitation"], latest["wind_speed_10m"], latest["pressure_msl"]]]
        pred = float(st.session_state.model.predict(X_latest)[0])
        pred_box.metric("Predicted temperature_2m", f"{pred:.2f} °C")

    df_box.dataframe(st.session_state.df.tail(20), use_container_width=True)
    time.sleep(n)
