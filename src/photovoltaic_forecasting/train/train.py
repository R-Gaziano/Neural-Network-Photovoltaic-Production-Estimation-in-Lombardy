"""

We train the LSTM.

"""

import numpy as np
import joblib
import torch
import torch.nn as nn


def train(Y: np.array, X: np.array, model, scaler, pca, config: dict):

    #Import number of componets for PCA
    
    learning_rate = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 3. Move the model and the split dataset to the GPU
     #24

    X, Y = X.to(device), Y.to(device)

    # 4. Standard training loop (updated to use the training set)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Train only on the training split
        predictions = model(X)
        loss = criterion(predictions, Y)
        
        loss.backward()
        optimizer.step()
        
        # Optional: Print progress and evaluate on test set every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval() # Switch to evaluation mode
            
            print(f'Epoch {epoch+1:3d} | Train Loss: {loss.item():.4f}')



    # 1. Save the PyTorch model weights
    torch.save(model.state_dict(), "./artifacts/lstm_model_weights.pth")

    # 2. Save the preprocessors
    joblib.dump(scaler, "./artifacts/scaler.joblib")
    joblib.dump(pca, "./artifacts/pca.joblib")

    print("Model and preprocessors saved successfully!")