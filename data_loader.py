import torch
from torch.utils.data import Dataset, DataLoader

class BowDataset(Dataset):

    def __init__(self, x, y, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = x
        self.y = torch.tensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        #sample = {'x': self.x[idx], 'y': self.y[idx]}
        sample = (torch.tensor(self.x[idx]), self.y[idx])
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
    scores = torch.stack(scores, 0)

    # Pad answer
    lengths = [len(answer) for answer in answers]
    padded = torch.zeros(len(answers), max(lengths)).long()
    for i, answer in enumerate(answers):
        end = lengths[i]
        padded[i, :end] = answer[:end]

    return padded, scores, lengths