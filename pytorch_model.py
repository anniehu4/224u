from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn import metrics

class BowDataset(Dataset):

    def __init__(self, x, y, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
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

class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    y_true = []
    y_pred = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device).float()
        y_true += target.numpy().tolist()
        optimizer.zero_grad()
        output = model(data)
        y_pred += output.cpu().detach().numpy().tolist()

        mse = nn.MSELoss()
        loss = mse(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    epoch_r2 = metrics.r2_score(y_true, y_pred)
    print('Train Epoch: {} Train r2: {}'.format(
                epoch, epoch_r2))

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

def dnn(x_train, y_train, x_test, y_test):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
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
    train_loader = DataLoader(
        BowDataset(x_train, y_train),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = DataLoader(
        BowDataset(x_test, y_test),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    n_features = x_train.shape[1]
    model = Net(n_features).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)


#if __name__ == '__main__':
#    main()
