from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn import metrics
from data_loader import *
from utils import pad

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x, lengths=None):
        x = F.relu(self.fc1(x))
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

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    y_true = []
    y_pred = []
    for batch_idx, (data, target, lengths) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device).float()
        y_true += target.numpy().tolist()

        output = model(data, lengths)
        y_pred += output.cpu().detach().numpy().tolist()

        mse = nn.MSELoss()
        loss = mse(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    epoch_r2 = metrics.r2_score(y_true, y_pred)
    print('Train Epoch: {} Train r2: {}'.format(
                epoch, epoch_r2))
    return y_true, y_pred

def test(args, model, device, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    test_loss = 0
    with torch.no_grad():
        for data, target, lengths in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            y_true += target.numpy().tolist()
            output = model(data, lengths)
            y_pred += output.cpu().detach().numpy().tolist()

            mse = nn.MSELoss()
            loss = mse(output, target)
            #test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            test_loss += loss


    test_loss /= len(test_loader)
    epoch_r2 = metrics.r2_score(y_true, y_pred)
    print('\nTest set: Average loss: {:.4f}, Test r2: {})\n'.format(
        test_loss, epoch_r2))
    return y_true, y_pred

def get_optimizer(optimizer_type, model):
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    return optimizer

def basic_nn(x_train, y_train, x_test, y_test):
    # Training settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(
        BowDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    test_loader = DataLoader(
        BowDataset(x_test, y_test),
        batch_size=args.test_batch_size, shuffle=True, collate_fn=collate_fn)

    rnn = False
    optimizer_type = 'adam'

    n_features = x_train.shape[1]

    model = Net(n_features).to(device)
    optimizer = get_optimizer(optimizer_type, model)

    for epoch in range(1, args.epochs + 1):
        train_true, train_pred = train(args, model, device, train_loader, optimizer, epoch)
        test_true, test_pred = test(args, model, device, test_loader)

    return (train_true, train_pred, test_true, test_pred)


def rnn(x_train, y_train, x_test, y_test):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(
        BowDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)
    
    test_loader = DataLoader(
        BowDataset(x_test, y_test),
        batch_size=args.test_batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)

    optimizer_type = 'sgd'

    embed_size = 200
    hidden_size = 100

    model = RNN(embed_size, hidden_size, 2).to(device)

    optimizer = get_optimizer(optimizer_type, model)

    for epoch in range(1, args.epochs + 1):
        train_true, train_pred = train(args, model, device, train_loader, optimizer, epoch)
        test_true, test_pred = test(args, model, device, test_loader)

    return (train_true, train_pred, test_true, test_pred)