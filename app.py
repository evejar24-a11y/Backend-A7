import time
import pandas as pd
import streamlit as st
from backend import fetch_row, update_model

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
