from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn import metrics
from utils import pad

class BowDataset(Dataset):

    def __init__(self, x, y, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        #sample = {'x': self.x[idx], 'y': self.y[idx]}
        sample = (self.x[idx], 
            self.y[idx])
        return sample

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (features, score).

    We should build custom collate_fn rather than using default collate_fn,
    because padding is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - answer: torch tensor of shape (?); variable length
            - score: torch tensor of shape (1,)
    Returns:
        answers: torch tensor of shape (batch_size, padded_length).
        scores: torch tensor of shape (batch_size, 1).
        lengths: list; valid length for each padded answer.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[0]), reverse=True)
    answers, scores = zip(*data)

    # Merge scores (from tuple of 1D tensor to 2D tensor).
    scores = torch.stack(scores, 0)

    # Merge answers (from tuple of 2D tensor to 3D tensor).
    lengths = [len(answer) for answer in answers]
    padded = torch.zeros(len(answers), max(lengths)).long()
    for i, answer in enumerate(answers):
        end = lengths[i]
        padded[i, :end] = answer[:end]
    return padded, scores, lengths

class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # "nn.Linear" performs the last step of prediction, where it maps the hidden layer to # classes
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    y_true = []
    y_pred = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device).float()
        y_true += target.numpy().tolist()

        output = model(data)
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
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            y_true += target.numpy().tolist()
            output = model(data)
            y_pred += output.cpu().detach().numpy().tolist()

            mse = nn.MSELoss()
            loss = mse(output, target)
            #test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            test_loss += loss


    test_loss /= len(test_loader.dataset)
    epoch_r2 = metrics.r2_score(y_true, y_pred)
    print('\nTest set: Average loss: {:.4f}, Test r2: {})\n'.format(
        test_loss, epoch_r2))
    return y_true, y_pred

def get_optimizer(optimizer_type, model):
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    return optimizer

def dnn(x_train, y_train, x_test, y_test):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if rnn:
        x_train, train_lengths = pad(x_train, max_len=500, dim=1)
        x_test, test_lengths = pad(x_test, max_len=500, dim=1)
        x_train = x_train.unsqueeze(2)
        x_test = x_test.unsqueeze(2)

    train_loader = DataLoader(
        BowDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)
    
    test_loader = DataLoader(
        BowDataset(x_test, y_test),
        batch_size=args.test_batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)

    rnn = True
    optimizer_type = 'adam'

    n_features = x_train.shape[1]
    embed_size = 50
    hidden_size = 20

    if rnn:
        # x_train = x_train.reshape((x_train.shape[0], -1, embed_size))
        # x_test = x_test.reshape((x_test.shape[0], -1, embed_size))
        model = RNN(1, hidden_size, x_train.shape[1]).to(device)
    else:
        model = Net(n_features).to(device)

    optimizer = get_optimizer(optimizer_type, model)

    for epoch in range(1, args.epochs + 1):
        train_true, train_pred = train(args, model, device, train_loader, optimizer, epoch)
        test_true, test_pred = test(args, model, device, test_loader)

    return (train_true, train_pred, test_true, test_pred)


#if __name__ == '__main__':
#    main()
