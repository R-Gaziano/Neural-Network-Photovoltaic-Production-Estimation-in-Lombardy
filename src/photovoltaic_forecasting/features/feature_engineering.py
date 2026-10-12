"""

The file is used for feature engineering of the dataframe.
Returns Y, X, scaler, pca.

"""

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib


def create_sliding_windows_training(data_x, data_y, seq_length):
        xs, ys = [], []

        for i in range(len(data_x) - seq_length):
            # Grab a window of 'seq_length' rows
            xs.append(data_x[i:(i + seq_length)])
            # Grab the target for the next time step
            ys.append(data_y[i + seq_length])   

        X = torch.tensor(np.array(xs), dtype=torch.float32)
        Y = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)
            
        return X, Y


def create_sliding_windows_prediction(data_x, seq_length):
        xs, ys = [], []
        # Stop before we run out of sequence length
        for i in range(len(data_x) - seq_length):
            # Grab a window of 'seq_length' rows
            xs.append(data_x[i:(i + seq_length)])
            # Grab the target for the next time step
            
        return np.array(xs)


def Feature_engineering(dataframe: pd.DataFrame, config: dict, prediction = False) -> pd.DataFrame:


    

    if prediction == False:

    #Import number of componets for PCA
        n_components = config["feature_engineering"]["PCA_components"]
    
        #Set target and features
        X = dataframe.drop(columns=["Renewable Generation", "Data"]).to_numpy()
    
    
    
        ## Dimensionality Reduction ##
    
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
    
        # PCA on the scaled data
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X_scaled)
        
    
    
    
        ## Feature engineering to add day-night info for the model ##
    
        # Start with the train data:
        hour_sin = np.sin(2 * np.pi * pd.to_datetime(dataframe["Data"]).dt.hour / 24.0)
        hour_cos = np.cos(2 * np.pi *  pd.to_datetime(dataframe["Data"]).dt.hour / 24.0)
    
        # Combine features
        X_combined = np.column_stack((X, hour_sin, hour_cos))
    
        # Create sequences
        seq_length = 24

        Y = dataframe["Renewable Generation"].to_numpy()

        X, Y = create_sliding_windows_training(X_combined, Y, seq_length)
        
        return Y, X, scaler, pca

    else:
        numeric_df = dataframe.drop(columns=["Data"])

        # Load sklearn artifacts
        scaler = joblib.load("./artifacts/scaler.joblib")
        pca = joblib.load("./artifacts/pca.joblib")

        n_components = 20

        new_X_scaled = scaler.transform(numeric_df)
        new_X_pca = pca.transform(new_X_scaled)

        hour_sin = np.sin(2 * np.pi * pd.to_datetime(dataframe["Data"], format="%d/%m/%Y %H:%M:%S").dt.hour / 24.0)
        hour_cos = np.cos(2 * np.pi *  pd.to_datetime(dataframe["Data"], format="%d/%m/%Y %H:%M:%S").dt.hour / 24.0)

        # 1. Combine features (Result is 2D: samples, 3)
        X_combined = np.column_stack((new_X_pca, hour_sin, hour_cos))

        # 2. Create sequences (e.g., look back 24 hours)
        seq_length = 24
        X_seq = create_sliding_windows_prediction(X_combined, seq_length)

        # 3. Convert to 3D tensors (Now X is shape: samples, 24, 3)
        X_new = torch.tensor(X_seq, dtype=torch.float32)

        return X_new