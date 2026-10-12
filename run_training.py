"""

This file run the training pipeline of a LSTM model for estimating energy production given temporal trends and
metereological conditions.

"""


import yaml
import torch
from src.photovoltaic_forecasting.data.data_for_training import data_for_training_importing
from src.photovoltaic_forecasting.features.feature_engineering import Feature_engineering
from src.photovoltaic_forecasting.model.model import TimeSeriesLSTM
from src.photovoltaic_forecasting.train.train import train

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
    
    dataframe = data_for_training_importing()

    print("Data loaded...")

    Y, X, scaler, pca =  Feature_engineering(dataframe=dataframe, config = config)

    print("Transformation done...")

    model = TimeSeriesLSTM(input_size = n_components+2, hidden_size=hidden_size, output_size=1).to(device)

    print("Model set...")

    print("Starting training:")

    train(Y = Y, X = X, model = model, scaler = scaler, pca = pca, config = config)


    print("DONE!")

main()
