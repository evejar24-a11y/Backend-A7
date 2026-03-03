import time
import requests
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

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
