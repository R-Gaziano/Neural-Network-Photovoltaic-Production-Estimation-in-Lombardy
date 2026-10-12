
"""
Download temperature and radiation data for model prediction.
Returns a pandas DataFrame.
"""

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter


URL = (
    "https://iris.arpalombardia.it/gisINM/"
    "open-scripts/plot_rete_iris_lm_guest.php"
)

# Each worker thread gets its own session, allowing connection reuse.
_thread_local = threading.local()


def get_session():
    if not hasattr(_thread_local, "session"):
        session = requests.Session()

        adapter = HTTPAdapter(
            pool_connections=2,
            pool_maxsize=1,
        )
        session.mount("https://", adapter)

        _thread_local.session = session

    return _thread_local.session


def download_data(sensor, sensor_type="T"):
    params = {
        "codice_istat": "620",
        "id_parametro": f"{sensor_type}|{sensor}",
    }

    response = get_session().get(
        URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    match = re.search(
        r"graph_data\s*=\s*(\[.*?\]);",
        response.text,
        re.S,
    )

    if not match:
        raise RuntimeError(
            f"graph_data not found for sensor {sensor} "
            f"(type={sensor_type})"
        )

    data = json.loads(match.group(1))

    df = pd.DataFrame(data, columns=["timestamp", "value"])

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    return df


def fetch(sensor_id, sensor_type="T"):
    df = download_data(sensor_id, sensor_type)

    series = (
        df.set_index("datetime")["value"]
        .rename(sensor_id)
    )

    return sensor_id, series


def data_for_prediction_importing(max_workers=8) -> pd.DataFrame:

    sensors_list_temperature = pd.read_csv(
        "./src/photovoltaic_forecasting/utils/"
        "temperature_sensors_list"
    )["ID_sensori"]

    sensors_list_radiation = pd.read_csv(
        "./src/photovoltaic_forecasting/utils/"
        "radiation_sensors_list"
    )["ID_sensori"]

    # Submit BOTH sensor groups to the same thread pool.
    tasks = [
        (sensor, "T") for sensor in sensors_list_temperature
    ] + [
        (sensor, "RG") for sensor in sensors_list_radiation
    ]

    total = len(tasks)
    results_temperature = {}
    results_radiation = {}

    print("Download started...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        future_to_task = {
            executor.submit(fetch, sensor, sensor_type): (
                sensor,
                sensor_type,
            )
            for sensor, sensor_type in tasks
        }

        for completed, future in enumerate(
            as_completed(future_to_task), start=1
        ):
            sensor, sensor_type = future_to_task[future]
            sensor_id, series = future.result()

            if sensor_type == "T":
                results_temperature[sensor_id] = series
            else:
                results_radiation[sensor_id] = series

            print(
                f"Download: {completed / total:.1%}",
                end="\r",
            )

    print("Download completed.      ")

    # Concatenate in the original sensor order.
    temperature = pd.concat(
        [results_temperature[i] for i in sensors_list_temperature],
        axis=1,
    )

    temperature = temperature.reset_index().iloc[::-1]
    temperature_data = temperature.iloc[3:].copy()

    radiation = pd.concat(
        [results_radiation[i] for i in sensors_list_radiation],
        axis=1,
    )

    radiation = radiation.reset_index().iloc[::-1]
    radiation_data = radiation.iloc[3:].copy()

    # Preserve the original inner-join behavior.
    prediction_data = pd.merge(
        temperature_data,
        radiation_data,
        on="datetime",
    )

    prediction_data.columns = prediction_data.columns.astype(str)

    # datetime is already a datetime column; no need to convert it again.
    prediction_data["Data"] = prediction_data["datetime"].dt.strftime(
        "%d/%m/%Y %H:%M:%S"
    )

    prediction_data = prediction_data.drop(columns="datetime")

    return prediction_data
