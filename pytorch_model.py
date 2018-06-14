from __future__ import print_function
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from model.architecture import Net, RNN
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn import metrics
from data_loader import *
from utils import pad

"""
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
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
"""

BATCH_SIZE = 64
TEST_BATCH_SIZE = 10
EPOCHS = 30
LR = 0.001
MOMENTUM = 0.5
NO_CUDA = False
SEED = 1
LOG_INTERVAL = 50


def train_rnn(model, device, train_loader, optimizer, epoch):
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
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    epoch_r2 = metrics.r2_score(y_true, y_pred)
    print('Train Epoch: {} Train r2: {}'.format(
                epoch, epoch_r2))
    return y_true, y_pred

def train(model, device, train_loader, optimizer, criterion, epoch, classify):
    model.train()
    y_true = []
    y_pred = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device).float()
        y_true += np.reshape(target.numpy(), -1).tolist()

        output, embed = model(data, None)
        y_pred += np.reshape(output.cpu().detach().numpy(), -1).tolist()

        mse = nn.MSELoss()
        loss = mse(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


    if classify:
        y_true = np.array(y_true)
        y_pred = np.array([1.0 if pred > 0.5 else 0.0 for pred in y_pred])
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred)

        print('Train Epoch: {} Train acc: {}'.format(
                epoch, 100. * np.mean(y_true == y_pred)))

    else:
        epoch_r2 = metrics.r2_score(y_true, y_pred)
        print('Train Epoch: {} Train r2: {}'.format(
                epoch, epoch_r2))

    return y_true, y_pred

def test_rnn(model, device, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    test_loss = 0
    with torch.no_grad():
        for data, target, lengths in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            y_true += np.reshape(target.numpy(), -1).tolist()
            output = model(data, lengths)
            y_pred += np.reshape(output.cpu().detach().numpy(), -1).tolist()

            mse = nn.MSELoss()
            loss = mse(output, target)
            #test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            test_loss += loss


    test_loss /= len(test_loader)
    epoch_r2 = metrics.r2_score(y_true, y_pred)
    print('\nTest set: Average loss: {:.4f}, Test r2: {})\n'.format(
        test_loss, epoch_r2))
    return y_true, y_pred

def test(model, device, test_loader, criterion, classify):
    model.eval()
    y_true = []
    y_pred = []
    embeds = []
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            y_true += np.reshape(target.numpy(), -1).tolist()
            output, embed = model(data, None)
            embeds += embed.cpu().detach().numpy().tolist()
            y_pred += np.reshape(output.cpu().detach().numpy(), -1).tolist()

            loss = criterion(output, target)
            #test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            test_loss += loss


    test_loss /= len(test_loader)
    if classify:
        y_true = np.array(y_true)
        y_pred = np.array([1.0 if pred > 0.5 else 0.0 for pred in y_pred])
        acc = 100. * np.mean(y_true == y_pred)
        metric = acc
        print('Test loss: {} Test acc: {}'.format(test_loss, acc))
    else:
        epoch_r2 = metrics.r2_score(y_true, y_pred)
        metric = epoch_r2
        print('Test loss: {} Test r2: {}'.format(test_loss, epoch_r2))

    return y_true, y_pred, metric, embeds

def get_optimizer(optimizer_type, model):
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    return optimizer

def get_criterion(classify):
    if classify:
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss() 
    return criterion
    

def basic_nn(x_train, y_train, x_test, y_test, pretrained=False, classify=False):
    # Training settings
    use_cuda = not NO_CUDA and torch.cuda.is_available()

    torch.manual_seed(SEED)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(
        BowDataset(x_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True)
    
    test_loader = DataLoader(
        BowDataset(x_test, y_test),
        batch_size=TEST_BATCH_SIZE, shuffle=True)

    optimizer_type = 'adam'

    n_features = x_train.shape[1]

    path = 'model/nn-classify.ckpt' if classify else 'model/nn-reg.ckpt'
    model = Net(n_features, classify).to(device)
    optimizer = get_optimizer(optimizer_type, model)
    criterion = get_criterion(classify)

    if pretrained:
        print("=> loading model checkpoint")
        model.load_state_dict(torch.load(path))
        test_true, test_pred, metric, embeds = test(model, device, test_loader, criterion, classify)
        if classify:
            pickle.dump((test_true, test_pred, embeds), open('data/learned_embeds.pkl', 'wb'))
        return ([], [], test_true, test_pred)

    best_metric = float("-inf")
    best_epoch = 0
    best_test_true = []
    best_test_pred = []
    best_train_true = []
    best_train_pred = []

    for epoch in range(1, EPOCHS + 1):
        print("===================")
        train_true, train_pred = train(model, device, train_loader, optimizer, criterion, epoch, classify)
        test_true, test_pred, metric, embeds = test(model, device, test_loader, criterion, classify)
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_test_true = test_true
            best_test_pred = test_pred
            best_train_true = train_true
            best_train_pred = train_pred
            torch.save(model.state_dict(), path)
            if classify:
                pickle.dump((test_true, test_pred, embeds), open('data/nn-learned_embeds.pkl', 'wb'))

    print("===================")
    if classify:
        print('\nBest Test acc: {} on epoch: {})\n'.format(best_metric, best_epoch))
    else:
        print('\nBest Test r2: {} on epoch: {})\n'.format(best_metric, best_epoch))
    return (train_true, train_pred, test_true, test_pred)


def rnn(x_train, y_train, x_test, y_test, pretrained=False, classify=False):
    print("x_train", len(x_train))
    print("y_train", y_train.shape)
    print("x_test", len(x_test))
    print("y_test", y_test.shape)
    use_cuda = not NO_CUDA and torch.cuda.is_available()

    torch.manual_seed(SEED)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = DataLoader(
        BowDataset(x_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, **kwargs)
    
    test_loader = DataLoader(
        BowDataset(x_test, y_test),
        batch_size=TEST_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, **kwargs)

    optimizer_type = 'sgd'

    embed_size = 200
    hidden_size = 100

    path = 'model/rnn-classify.ckpt' if classify else 'model/rnn-reg.ckpt'
    model = RNN(embed_size, hidden_size, 2).to(device)

    optimizer = get_optimizer(optimizer_type, model)

    if pretrained:
        print("=> loading model checkpoint")
        model.load_state_dict(torch.load(path))
        test_true, test_pred, metric, embeds = test(model, device, test_loader, criterion, classify)
        if classify:
            pickle.dump((test_true, test_pred, embeds), open('data/rnn-learned_embeds.pkl', 'wb'))
        return ([], [], test_true, test_pred)

    best_metric = float("-inf")
    best_epoch = 0
    best_test_true = []
    best_test_pred = []
    best_train_true = []
    best_train_pred = []

    for epoch in range(1, EPOCHS + 1):
        train_true, train_pred = train_rnn(model, device, train_loader, optimizer, epoch, False)
        test_true, test_pred, metric = test_rnn(model, device, test_loader, False)
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_test_true = test_true
            best_test_pred = test_pred
            best_train_true = train_true
            best_train_pred = train_pred
            torch.save(model.state_dict(), path)
            if classify:
                pickle.dump((test_true, test_pred, embeds), open('data/nn-learned_embeds.pkl', 'wb'))

    print("===================")
    if classify:
        print('\nBest Test acc: {} on epoch: {})\n'.format(best_metric, best_epoch))
    else:
        print('\nBest Test r2: {} on epoch: {})\n'.format(best_metric, best_epoch))
    return (train_true, train_pred, test_true, test_pred)