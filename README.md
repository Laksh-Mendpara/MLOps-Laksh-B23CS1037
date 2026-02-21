# ML-DLOps Minor Exam – SET A

**Name:** Mendpara Daksh Alpeshbhai  
**Roll No:** B23CS1037  
**Branch:** Minor_Exam

---

## Q1. [Docker-A] – Digit Classification with ResNet-18

### Commands

**Build Image:**
```bash
docker build --network=host -t minor_exam:v1 .
```

**Run evaluate.py inside container:**
```bash
docker run -it --rm --gpus all --shm-size=8g -v $(pwd):/workspace minor_exam:v1 \
  bash -c "cd /workspace && python evaluate.py"
```

**Second Container – manual setup + train.py:**
```bash
# Container creation:
docker run -d --name exam_train_manual --gpus all --network=host --shm-size=8g \
  -v $(pwd):/workspace pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
  bash -c "pip install matplotlib seaborn scikit-learn Pillow && cd /workspace && python train.py"

# Dependencies installation (inside container):
pip install matplotlib seaborn scikit-learn Pillow
```

### Results on Test Set (setA.pth)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **89.00%** |
| **F1 Score** | **0.8413** |
| **Class 5 Accuracy** | **0.00%** |
| `data/test/5/340.png` prediction | **Class 3** (Confidence: 47.65%) |

### Class-wise Accuracy

| Class | Accuracy |
|-------|----------|
| 0 | 99.80% |
| 1 | 99.30% |
| 2 | 98.45% |
| 3 | 97.82% |
| 4 | 95.11% |
| **5** | **0.00%** |
| 6 | 96.45% |
| 7 | 99.03% |
| 8 | 93.84% |
| 9 | 99.01% |

### Hyperparameter Analysis (Second Container – train.py)

**Settings used:** Pretrained ResNet-18, Adam (LR=1e-3), StepLR × 0.5 every 3 epochs, BS=32, 10 epochs, Augmentations (HFlip, RandomCrop).

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 1 | 0.2007 | 93.60% |
| 5 | 0.0442 | 98.59% |
| 10 | 0.0184 | **99.38%** |

**Analysis:** Pretrained ImageNet weights gave a strong 93.6% at epoch 1. LR scheduler caused stable convergence to 99.38%. Augmentations (flip + crop) prevented overfitting effectively.

**Best Setting – Overall Accuracy:** 89.00% (on held-out test with `setA.pth`)  
**Best Setting – Class 5 Accuracy:** 0.00% (class underrepresented / model bias)

---

## Q2. [HuggingFace-A] – STL-10 Image Classification

**WandB Project:** [stl10-classification](https://wandb.ai/b23cs1037-iit-jodhpur/stl10-classification)

### Setup
- Dataset: [Chiranjeev007/STL-10_Subset](https://huggingface.co/datasets/Chiranjeev007/STL-10_Subset)
- Model: Pretrained ResNet-18 → fc replaced for 10 classes
- Optimizer: Adam (LR=1e-4, WD=1e-4), CosineAnnealingLR (T_max=15)
- Batch Size: 32 | Epochs: 15
- Augmentation: RandomHFlip, RandomRotation±10°, ColorJitter

**Build & Run:**
```bash
docker build --network=host -t huggingface_a:v1 .

# Train
docker run -it --rm --gpus all --network=host --shm-size=8g \
  -v $(pwd):/workspace huggingface_a:v1 bash -c "cd /workspace && python train.py"

# Evaluate
docker run -it --rm --gpus all --network=host --shm-size=8g \
  -v $(pwd):/workspace huggingface_a:v1 bash -c "cd /workspace && python evaluate.py"
```

### Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **95.80%** ✅ (>75% requirement) |
| Best Val Accuracy | 95.00% |

### Class-wise Accuracy

| Class | Name | Accuracy |
|-------|------|----------|
| 0 | airplane | 97.00% |
| 1 | bird | 98.00% |
| 2 | car | 99.00% |
| 3 | cat | 93.00% |
| 4 | deer | 94.00% |
| 5 | dog | 89.00% |
| 6 | horse | 94.00% |
| 7 | monkey | 97.00% |
| 8 | ship | 98.00% |
| 9 | truck | 99.00% |

### WandB Logs
- Train/Val Loss & Accuracy curves (15 epochs)
- Confusion Matrix (test set)
- Class-wise Accuracy Bar Chart (with class names)
- 20 sample predictions (10 correct / 10 incorrect) with labels
