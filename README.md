# Neural-Network-Photovoltaic-Production-Estimation-in-Lombardy
This Repository uses a RNN (LSTM) to predict real time production of photovoltaic energy in Lombardy.

## Project Overview
The Terna publishes official photovoltaic production data only once a year, limiting the possibility to monitor real-time regional energy production. This project estimates current photovoltaic production in Lombardy, expressed in GWh, using real-time environmental measurements (solar radiation and temperature) from 331 sensors around the whole region. The system applies an object-oriented machine learning pipeline to process meteorological data, handle missing sensor values, and predict regional photovoltaic output using a Long Short-Term Memory (LSTM) neural network.
## Model Performance
The LSTM model is evaluated against a Seasonal Autoregressive Integrated Moving Average (SARIMA) baseline.

| Metric | LSTM Model | SARIMA Baseline |
| --- | --- | --- |
| **RMSE** | 0.019 GWh | 0.200 GWh |
| **R²** | 0.9458 | 0.8210 |

The LSTM achieves a lower Root Mean Squared Error (RMSE) and a higher coefficient of determination (R²) than the SARIMA baseline on the evaluated dataset, indicating that the LSTM captures the observed production patterns more effectively.

## Technical Architecture
The system consists of a model training workflow and a real-time prediction workflow, built with PyTorch and scikit-learn. 


* **Data Ingestion:** Retrieves temperature and global solar radiation measurements from ARPA. Concurrent threading is used to improve the efficiency of sensor data retrieval.


* **Spatial Interpolation (Kriging):** Handles missing sensor measurements using Ordinary Kriging. Sensor UTM coordinates and an exponential variogram model are used to estimate missing values from spatially related observations.


* **Feature Engineering:** Standardizes input features using `StandardScaler` and applies Principal Component Analysis (PCA) for dimensionality reduction. The hour of the day is encoded using sine and cosine transformations to represent cyclical daily patterns.


* **Sequence Generation:** Organizes the processed features into 24-hour sliding windows for input to the recurrent neural network.


* **Modeling:** Uses a custom `LSTM` model implemented in PyTorch to estimate photovoltaic production. Negative predictions are clipped to zero during inference to ensure physically meaningful output.



## Data Sources

The project combines two main data sources:

* **ARPA Lombardia:** Environmental sensor measurements, including temperature and global solar radiation.
* **Terna:** Historical photovoltaic production data used to train and evaluate the forecasting model.

*Note: Historical production CSV files must be placed in the `Data/` directory before training. No API credentials are required for the installation or execution commands documented below.*


## Installation

**Requirements**

* Python 3.14.7
* Poetry

Install the project dependencies from the repository root:

```
poetry install

```

**Dataset for training**

* The dataset for training is available on [Kaggle](https://www.kaggle.com/datasets/robgaz/photovoltaic-and-sensors-data-lombardy).
* Place it on:

```
./Data

```

## Usage

**1. Train the Model**
Run the training pipeline:

```
poetry run python run_training.py

```

The training pipeline processes the historical data, fits the preprocessing transformations, trains the LSTM model, and saves the resulting model artifacts.

**2. Generate Predictions**
Run the prediction pipeline:

```
poetry run python run_prediction.py

```

The prediction pipeline retrieves the latest environmental sensor measurements, handles missing values, applies the trained preprocessing transformations, and loads the trained LSTM model to estimate current photovoltaic production.

The generated predictions are saved to:
`predictions/output.csv`

## Technologies

* **Python:** Core programming language
* **PyTorch:** LSTM model implementation
* **scikit-learn:** Feature scaling and dimensionality reduction
* **pykrige:** Spatial interpolation and missing values filling
* **Poetry:** Dependency and environment management
* **ARPA Lombardia API:** Environmental sensor data
* **Terna:** Historical photovoltaic production data
