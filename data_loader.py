import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

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
        return len(self.x)

    def __getitem__(self, idx):
        #sample = {'x': self.x[idx], 'y': self.y[idx]}
        sample = (torch.tensor(self.x[idx]), torch.tensor(self.y[idx]))
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
    # data is list of (answer, score) pairs, len(data) = batch_size
    # where answer is tensor of size (seq_len, hidden_dim)

    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[0]), reverse=True)

    answers, scores = zip(*data)
    # answers is tuple of length batch_size
    # scores is tuple of length batch_size

    scores = torch.stack(scores, 0)
    # scores is torch tensor of size [64, 1]

    batch_size = len(answers)
    embed_sz = answers[0].size(1)

    # Pad answer
    lengths = np.array([len(answer) for answer in answers])
    padded = torch.zeros(batch_size, max(lengths), embed_sz).float()

    # padded is (batch_size, seq_len, embed_sz)
    for i, answer in enumerate(answers):
        end = lengths[i]
        padded[i, :end, :] = answer[:end, :]

    # padded is FloatTensor, size=(batch_size, seq_len, embed_sz)
    # scores is DoubleTensor, size=(batch_size, 1)
    # lengths is numpy array, size=(batch_size)
    return (padded, scores, lengths)