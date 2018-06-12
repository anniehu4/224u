import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 100)
        self.fc2 = nn.Linear(100, 1)

        self.bn1 = nn.BatchNorm1d(100)

    def forward(self, x, lengths=None):
        x = F.relu(self.fc1(x))
        # TODO: can add batch_norm and dropout if wanted
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc1 = nn.Linear()
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x, lengths):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if self.input_size == 1:
            packed = pack_padded_sequence(x.unsqueeze(2), lengths, batch_first=True)
        else:
            packed = x.unsqueeze(1)
        
        # Forward propagate LSTM
        """
        print(type(x))
        print(x.type())
        print(x.size())
        print(x)
        print('-')
        print(type(packed))
        print(packed.type())
        print(packed.size())
        print(packed)
        print("-=========")
        """
        hidden, _ = self.lstm(packed, (h0, c0))
        if self.input_size == 1:
            hidden, _ = pad_packed_sequence(hidden, batch_first=True)
            print(hidden.data.shape)
            last_hidden = hidden[np.arange(x.size(0)), lengths, :]
        else:
            last_hidden = hidden[:, -1, :]
        # Decode the hidden state of the last time step
        out = self.fc(last_hidden)
        return out