"""

As model is used a LSTM.

"""

import torch
import torch.nn as nn

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # 1. Change nn.RNN to nn.LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 2. Initialize BOTH hidden state (h0) and cell state (c0)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # 3. Pass the tuple (h0, c0) to the lstm layer
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :]) 
        return out