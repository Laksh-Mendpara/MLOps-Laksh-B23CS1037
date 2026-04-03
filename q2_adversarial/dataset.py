import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_dataloaders(config):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True, transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, train_dataset.classes

def get_subset_loader(dataset, num_samples, batch_size=32):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)
