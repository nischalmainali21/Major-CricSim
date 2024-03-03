from django.db import models
import torch.nn as nn

class MyLSTMWithSoftmax(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyLSTMWithSoftmax, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Extract the output of the last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        # out = F.softmax(out, dim=1)  # Apply softmax activation
        return out
