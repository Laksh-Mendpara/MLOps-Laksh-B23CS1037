import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from PIL import Image

# ==========================
# Config
# ==========================
DATA_DIR = "data/test/"
MODEL_PATH = "setA.pth"
BATCH_SIZE = 32
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset
# ==========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = dataset.classes
print("Classes:", class_names)

# ==========================
# Load Model
# ==========================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Fix: actually load the weights
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

print("Model Loaded Successfully!")

# ==========================
# Evaluation
# ==========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ==========================
# Overall Accuracy
# ==========================
overall_acc = accuracy_score(all_labels, all_preds)
print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")

# ==========================
# F1 Score
# ==========================
macro_f1 = f1_score(all_labels, all_preds, average='macro')
print(f"F1 Score: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ==========================
# Class-wise Accuracy
# ==========================
print("\nClass-wise Accuracy:")
for cls_idx, cls_name in enumerate(class_names):
    mask = (all_labels == cls_idx)
    if mask.sum() > 0:
        cls_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
        print(f"  Class {cls_idx} ({cls_name}): {cls_acc * 100:.2f}%")

# ==========================
# Confusion Matrix Plot
# ==========================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('True Label', fontsize=13)
plt.xlabel('Predicted Label', fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\nConfusion matrix saved to confusion_matrix.png")

# ==========================
# Predict Single Image
# ==========================
def predict_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    print(f"\nImage: {image_path}")
    print(f"Predicted Class: {class_names[pred.item()]}")
    print(f"Confidence: {confidence.item()*100:.2f}%")

# Evaluate specific image
image_path = "data/test/5/340.png"
predict_single_image(image_path)