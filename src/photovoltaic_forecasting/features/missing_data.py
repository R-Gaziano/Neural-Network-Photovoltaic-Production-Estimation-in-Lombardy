"""

For filling missing data is used a spatial model (Kriging) to interpolate the available data and do prediction.
If the data available is not sufficient (n>30) is used a temporal interpolation.

"""


import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
import warnings
from scipy.linalg import LinAlgWarning


def take_row(data, sensors_list):


    sensors_list = sensors_list[["ID_sensori", "UTM.Est", "UTM.Nord"]]


    df_values = data.tolist()
    df_values = data.drop(labels=["Data"]).to_numpy()
    sens = sensors_list["ID_sensori"].tolist()

    
    combined_data = pd.DataFrame({
        'ID_sensori': sens,
        'Value': df_values
    })

    row = pd.merge(combined_data, sensors_list, on = "ID_sensori")

    return row


def Kriging_NA(data: pd.DataFrame, config: dict) -> pd.DataFrame:

    variogram_model = config["missing_values"]["variogram_model"]

    #Extracting coordinates and values
    x_coords = data["UTM.Est"].values
    y_coords = data["UTM.Nord"].values
    z_values = pd.to_numeric(data["Value"], errors="coerce").to_numpy(dtype=float)

    #Create a mask
    valid_mask = ~np.isnan(z_values)
    missing_mask = np.isnan(z_values)

    #Apply masks
    x_valid = x_coords[valid_mask]
    y_valid = y_coords[valid_mask]
    z_valid = z_values[valid_mask]

    # Extract coordinates of the missing points
    x_missing = np.float64(x_coords[missing_mask])
    y_missing = np.float64(y_coords[missing_mask])

    #Kriging
    OK = OrdinaryKriging(
        x_valid, 
        y_valid, 
        z_valid,
        variogram_model = variogram_model #Exponential is set as default, is the best fit for spatial data of decaying natural events. 
        #Better than sperical which tends to everstimate boundary regions -> exp. more conservative there. Other than that the two are identical.
    )

    #Fit:
    z_est, sigmasq = OK.execute("points", x_missing, y_missing)

    #Sobstitute the data back using the mask
    z_values_complete = z_values.copy()
    z_values_complete[missing_mask] = np.asarray(z_est)

    data["Value"] = z_values_complete   

    return data



def fill_missing_data(dataframe: pd.DataFrame, config: dict) -> pd.DataFrame:

    warnings.filterwarnings(action='ignore', category=LinAlgWarning)
    
    sensors_list_temp = pd.read_csv("./src/photovoltaic_forecasting/utils/temperature_sensors_list")
    sensors_list_rad = pd.read_csv(
        "./src/photovoltaic_forecasting/utils/radiation_sensors_list"
    )

    sensors_list = pd.concat([sensors_list_temp,sensors_list_rad])

    for row in range(dataframe.shape[0]):

        dataframe_row = dataframe.iloc[row]
    

        dataframe_row = take_row(dataframe_row, sensors_list)

        dataframe_row_compleate = Kriging_NA(dataframe_row, config)

        cols = dataframe.columns.drop("Data")

        dataframe.loc[row, cols] = (
                dataframe_row_compleate["Value"].to_numpy()
            )

        print(
            f"Interpolation at {row / len(range(dataframe.shape[0])):.1%}", 
            end="\r",
            )

    return dataframe
