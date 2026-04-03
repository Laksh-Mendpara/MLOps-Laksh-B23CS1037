import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor

def get_dataloaders(config):
    # If token is None/empty, pass False to avoid using bad env tokens
    hf_token = config.HF_TOKEN if config.HF_TOKEN else False
    processor = ViTImageProcessor.from_pretrained(config.MODEL_NAME, token=hf_token)
    mean = processor.image_mean
    std = processor.image_std
    size = processor.size["height"]

    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = datasets.CIFAR100(
        root=config.DATA_DIR, train=True, download=True, transform=train_transform
    )
    
    test_dataset = datasets.CIFAR100(
        root=config.DATA_DIR, train=False, download=True, transform=test_transform
    )

    pin_memory = config.DEVICE == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, train_dataset.classes
