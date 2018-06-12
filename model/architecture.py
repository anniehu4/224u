import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_features, classify=False):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 100)
        self.fc2 = nn.Linear(100, 1)

        self.bn1 = nn.BatchNorm1d(100)
        self.classify = classify
        self.classify_fn = nn.Sigmoid()

    def forward(self, x, lengths=None):
        x = F.relu(self.fc1(x))
        # TODO: can add batch_norm and dropout if wanted
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        if self.classify:
            x = self.classify_fn(x)
        return x

class RNN(nn.Module):
    # input_size = embed_sz
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc1 = nn.Linear()
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x, lengths):
        # x.size() = (batch_size, max_seq_len, embed_sz)
        batch_size = x.size(0)
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        packed = pack_padded_sequence(x, lengths, batch_first=True)
        # Forward propagate LSTM
        output, _ = self.lstm(packed, (h0, c0))
        output, _ = pad_packed_sequence(output, batch_first=True)
        #  output size is (batch, seq_len, hidden_size * num_directions)
        last_hidden = output[np.arange(batch_size), lengths, :]
        # last_hidden size is (batch_size, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(last_hidden)
        return out