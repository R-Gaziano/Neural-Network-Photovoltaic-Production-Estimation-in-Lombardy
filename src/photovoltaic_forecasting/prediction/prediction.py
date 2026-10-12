import numpy as np
import joblib
import torch
import torch.nn as nn


def prediction(X: np.array, model):

    # ---------------------------------------------------------
    # 2. Load Preprocessors & Weights
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    model.load_state_dict(torch.load("./artifacts/lstm_model_weights.pth", map_location=device))
    model.to(device)
    model.eval()  # Disable dropout/batchnorm for inference

    # ---------------------------------------------------------
    # 3. Transform new_X & Predict
    # ---------------------------------------------------------


    # Convert to PyTorch Tensor
    input_tensor = X.to(device)

    # Run inference
    with torch.no_grad():
        raw_predictions = model(input_tensor)
        
    # Convert back to numpy array
    predictions = raw_predictions.cpu().numpy()

    for i in range(len(predictions)):
        if predictions[i] < 0:
            predictions[i] = 0
    
    print("Prediciton successfull!")
    return predictions