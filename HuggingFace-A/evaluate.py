import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import wandb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

# ==========================
# Configuration
# ==========================
HF_DATASET    = "Chiranjeev007/STL-10_Subset"
HF_TOKEN      = (os.environ.get("HF_TOKEN") or "").strip()
# Resolve full repo_id from token
_HF_REPO_NAME = "minorexam"
if HF_TOKEN:
    try:
        from huggingface_hub import HfApi as _HfApi
        _hf_username = _HfApi().whoami(token=HF_TOKEN)["name"]
        HF_REPO_ID   = f"{_hf_username}/{_HF_REPO_NAME}"
    except Exception:
        HF_REPO_ID   = _HF_REPO_NAME
else:
    HF_REPO_ID = _HF_REPO_NAME
WANDB_API_KEY = os.environ.get("wandb_token", "").strip()
WANDB_PROJECT  = "stl10-classification"
WANDB_RUN_NAME = "resnet18-evaluation"

BATCH_SIZE  = 32
NUM_CLASSES = 10
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
    config={"model": "resnet18", "dataset": HF_DATASET}
)

# ==========================
# Load Dataset
# ==========================
print("Loading dataset from HuggingFace...")
raw_dataset = load_dataset(HF_DATASET)

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


split_keys = list(raw_dataset.keys())
test_key   = "test" if "test" in split_keys else split_keys[-1]
test_dataset = STL10Dataset(raw_dataset[test_key], transform=val_test_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)

print(f"Test samples: {len(test_dataset)}")

# ==========================
# Load Model (from local best_model.pth or HF Hub)
# ==========================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

if HF_TOKEN and HF_REPO_ID:
    print(f"Loading model from HuggingFace Hub: {HF_REPO_ID}")
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="model.pth",
                                  token=HF_TOKEN)
else:
    model_path = "best_model.pth"
    print(f"Loading model from local path: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

# ==========================
# Inference
# ==========================
all_preds, all_labels, all_images_raw = [], [], []
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(DEVICE))
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        denorm = (images * std + mean).clamp(0, 1)
        all_images_raw.extend(denorm)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ==========================
# Overall Accuracy
# ==========================
overall_acc = accuracy_score(all_labels, all_preds) * 100
print(f"\nTest Accuracy: {overall_acc:.2f}%")
wandb.summary["test_accuracy"] = overall_acc

# ==========================
# Class-wise Accuracy
# ==========================
print("\nClass-wise Accuracy:")
cls_accuracies = []
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    mask = (all_labels == cls_idx)
    if mask.sum() > 0:
        cls_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum() * 100
        print(f"  Class {cls_idx} ({cls_name}): {cls_acc:.2f}%")
        wandb.summary[f"test_acc_{cls_name}"] = cls_acc
    else:
        cls_acc = 0.0
    cls_accuracies.append(cls_acc)

# ==========================
# Confusion Matrix
# ==========================
cm = confusion_matrix(all_labels, all_preds)
fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax_cm)
ax_cm.set_title('Confusion Matrix – Test Set', fontsize=16)
ax_cm.set_ylabel('True Label', fontsize=13)
ax_cm.set_xlabel('Predicted Label', fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
fig_cm.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
wandb.log({"confusion_matrix": wandb.Image(fig_cm, caption="Confusion Matrix – Test Set")})
plt.close(fig_cm)
print("Confusion matrix saved and logged to WandB.")

# ==========================
# Class-wise Accuracy Bar Plot
# ==========================
fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
bars = ax_bar.bar(CLASS_NAMES, cls_accuracies, color='steelblue', edgecolor='white')
ax_bar.set_xlabel("Class", fontsize=12)
ax_bar.set_ylabel("Accuracy (%)", fontsize=12)
ax_bar.set_title("Class-wise Accuracy on Test Set", fontsize=15)
ax_bar.set_ylim(0, 115)
for bar, acc in zip(bars, cls_accuracies):
    ax_bar.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
fig_bar.savefig("classwise_accuracy.png", dpi=150, bbox_inches='tight')
wandb.log({"classwise_accuracy_bar": wandb.Image(fig_bar, caption="Class-wise Accuracy")})
plt.close(fig_bar)
print("Class-wise accuracy bar chart saved and logged to WandB.")

# ==========================
# 20 Sample Predictions (10 correct / 10 incorrect)
# ==========================
correct_idx   = np.where(all_preds == all_labels)[0]
incorrect_idx = np.where(all_preds != all_labels)[0]
np.random.seed(42)
chosen = np.concatenate([
    np.random.choice(correct_idx,   min(10, len(correct_idx)),   replace=False),
    np.random.choice(incorrect_idx, min(10, len(incorrect_idx)), replace=False),
])

wandb_images = []
for idx in chosen:
    img_np  = (all_images_raw[idx].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    status  = "✓ Correct" if all_preds[idx] == all_labels[idx] else "✗ Wrong"
    caption = f"{status} | Pred: {CLASS_NAMES[all_preds[idx]]} | True: {CLASS_NAMES[all_labels[idx]]}"
    wandb_images.append(wandb.Image(img_np, caption=caption))

wandb.log({"test_samples": wandb_images})
print(f"Logged {len(wandb_images)} sample images to WandB.")

wandb.finish()
print("\nEvaluation complete!")
