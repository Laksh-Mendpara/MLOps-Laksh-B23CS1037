import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import wandb
from datasets import load_dataset
from huggingface_hub import HfApi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

# ==========================
# Configuration
# ==========================
HF_DATASET    = "Chiranjeev007/STL-10_Subset"
HF_TOKEN      = (os.environ.get("HF_TOKEN") or "").strip()
# Resolve full repo_id: auto-fetch username from token
_HF_REPO_NAME = "minorexam"
if HF_TOKEN:
    try:
        _hf_username = HfApi().whoami(token=HF_TOKEN)["name"]
        HF_REPO_ID   = f"{_hf_username}/{_HF_REPO_NAME}"
    except Exception:
        HF_REPO_ID   = _HF_REPO_NAME
else:
    HF_REPO_ID = _HF_REPO_NAME
WANDB_API_KEY = os.environ.get("wandb_token", "").strip()
WANDB_PROJECT = "stl10-classification"
WANDB_RUN_NAME = "resnet18-pretrained"

BATCH_SIZE  = 32
NUM_CLASSES = 10
EPOCHS      = 15
LR          = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["airplane", "bird", "car", "cat", "deer",
               "dog", "horse", "monkey", "ship", "truck"]

print(f"Using device: {DEVICE}")

# ==========================
# WandB Init
# ==========================
if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_RUN_NAME,
    config={
        "epochs":       EPOCHS,
        "batch_size":   BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "model":        "resnet18-pretrained",
        "dataset":      HF_DATASET,
    }
)

print("Loading dataset from HuggingFace...")
raw_dataset = load_dataset(HF_DATASET)
print("Dataset splits:", list(raw_dataset.keys()))

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class STL10Dataset(Dataset):
    def __init__(self, hf_split, transform=None):
        self.data      = hf_split
        self.transform = transform
        first_item     = self.data[0]
        self.img_col   = "image" if "image" in first_item else list(first_item.keys())[0]
        self.lbl_col   = "label" if "label" in first_item else list(first_item.keys())[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        image = item[self.img_col]
        label = item[self.lbl_col]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Identify split keys
split_keys = list(raw_dataset.keys())
print(f"split keys - {split_keys}")
train_key  = "train"
val_key    = "validation"
test_key   = "test"

train_dataset = STL10Dataset(raw_dataset[train_key], transform=train_transform)
val_dataset   = STL10Dataset(raw_dataset[val_key],   transform=val_test_transform)
test_dataset  = STL10Dataset(raw_dataset[test_key],  transform=val_test_transform)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ==========================
# Model
# ==========================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ==========================
# Loss / Optimizer / Scheduler
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==========================
# Training Loop
# ==========================
best_val_acc    = 0.0
best_model_path = "best_model.pth"

for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item()
        _, predicted   = torch.max(outputs, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    scheduler.step()

    avg_train_loss = train_loss / len(train_loader)
    train_acc      = 100.0 * train_correct / train_total

    # ---- Validation ----
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs  = model(images)
            loss     = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc      = 100.0 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    wandb.log({
        "epoch":      epoch + 1,
        "train_loss": avg_train_loss,
        "train_acc":  train_acc,
        "val_loss":   avg_val_loss,
        "val_acc":    val_acc,
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  → Best model saved (Val Acc: {best_val_acc:.2f}%)")

print(f"\nTraining complete! Best Val Acc: {best_val_acc:.2f}%")

# ==========================
# Push Best Model to HuggingFace Hub
# ==========================
if HF_TOKEN:
    print("\nPushing best model to HuggingFace Hub...")
    api = HfApi()
    try:
        api.create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, exist_ok=True)
    except Exception as e:
        print(f"Repo creation notice: {e}")

    model_card = f"""---
license: apache-2.0
tags:
- image-classification
- resnet18
- stl10
---

# STL-10 ResNet-18 Classifier

Fine-tuned ResNet-18 on STL-10 Subset.
- **Best Val Accuracy**: {best_val_acc:.2f}%
- **Classes**: {CLASS_NAMES}
"""
    with open("README.md", "w") as f:
        f.write(model_card)

    api.upload_file(path_or_fileobj=best_model_path, path_in_repo="model.pth",
                    repo_id=HF_REPO_ID, token=HF_TOKEN)
    api.upload_file(path_or_fileobj="README.md", path_in_repo="README.md",
                    repo_id=HF_REPO_ID, token=HF_TOKEN)
    print(f"Model pushed to https://huggingface.co/{HF_REPO_ID}")
else:
    print("\nHF_TOKEN not set – skipping HuggingFace Hub push.")
    print(f"Best model saved locally at: {best_model_path}")

wandb.finish()
