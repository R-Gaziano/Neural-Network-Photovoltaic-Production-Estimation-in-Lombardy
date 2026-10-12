"""

This file run the training pipeline of a LSTM model for estimating energy production given temporal trends and
metereological conditions.

"""

import pandas as pd
import torch
import yaml
from src.photovoltaic_forecasting.data.data_for_prediction import data_for_prediction_importing
from src.photovoltaic_forecasting.features.missing_data import fill_missing_data
from src.photovoltaic_forecasting.features.feature_engineering import Feature_engineering
from src.photovoltaic_forecasting.model.model import TimeSeriesLSTM
from src.photovoltaic_forecasting.prediction.prediction import prediction

print("Starting...")

def load_config(config_path: str = "configs/training_model_config.yaml") -> dict:
    """Loads the central configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config()

    n_components = config["feature_engineering"]["PCA_components"]
    
    hidden_size = config["model"]["hidden_size"]

    print("Config loaded...")
    
    dataframe = data_for_prediction_importing()

    print("Data downloaded...")

    dataframe = fill_missing_data(dataframe, config = config)

    print("Missing values filled...")

    X =  Feature_engineering(dataframe=dataframe, config = config, prediction=True)

    print("Transformation done...")

    model = TimeSeriesLSTM(input_size = n_components+2, hidden_size=hidden_size, output_size=1).to(device)

    print("Model set...")

    print("Starting prediction:")

    output = prediction(X = X, model = model)

    output = pd.DataFrame(output)

    print("DONE!")
    print("Prediction saved in predictions")

    output.to_csv("./predictions/output.csv")



main()
