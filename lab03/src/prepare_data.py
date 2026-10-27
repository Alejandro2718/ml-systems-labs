import torch
from torch.utils.data import TensorDataset, Subset, DataLoader
from torchvision import datasets, transforms

from common import DATA, SEED


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))
    ])
    train = datasets.FashionMNIST(DATA, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(DATA, train=False, download=True, transform=transform)
    assert len(train) == 60000 and len(test) == 10000
    def cache(dataset):
        batches = list(DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=0))
        return TensorDataset(torch.cat([x for x, _ in batches]),
                             torch.cat([y for _, y in batches]))
    train, test = cache(train), cache(test)
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(SEED))
    train_ids, val_ids = order[:55000], order[55000:]
    return Subset(train, train_ids.tolist()), Subset(train, val_ids.tolist()), test
