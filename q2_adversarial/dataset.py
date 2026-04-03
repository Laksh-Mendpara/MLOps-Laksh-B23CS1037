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
    eval_train_dataset = datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True, transform=test_transform
    )
    test_dataset = datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True, transform=test_transform
    )

    val_size = 5000
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    permutation = torch.randperm(len(train_dataset), generator=generator).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(eval_train_dataset, val_indices)

    loader_kwargs = {
        "num_workers": config.NUM_WORKERS,
        "pin_memory": config.DEVICE.startswith("cuda"),
        "persistent_workers": config.NUM_WORKERS > 0,
    }

    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader, train_dataset.classes

def get_subset_loader(dataset, num_samples, batch_size=32):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)
