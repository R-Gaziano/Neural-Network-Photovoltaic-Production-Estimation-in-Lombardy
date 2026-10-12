"""

This file is used to import the data for model training.
Return a pandas dataframe.

"""

import pandas as pd

def data_for_training_importing() -> pd.DataFrame:

    training_data = pd.read_csv("./Data/Dataframe_for_training.csv")

    return training_data

