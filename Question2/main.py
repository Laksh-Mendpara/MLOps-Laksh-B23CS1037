import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Local imports
from utils import MyDataset
from mymodel import UNet

# --- FAST VECTORIZED METRICS ---
def get_fast_metrics(preds, targets, num_classes=23):
    preds = torch.argmax(preds, dim=1).flatten()
    targets = targets.flatten()
    
    # Mask to ignore potential background/padding if necessary
    mask = (targets >= 0) & (targets < num_classes)
    hist = torch.bincount(
        num_classes * targets[mask].to(torch.int64) + preds[mask], 
        minlength=num_classes**2
    ).view(num_classes, num_classes)
    
    diag = torch.diag(hist)
    row_sum = hist.sum(1)
    col_sum = hist.sum(0)
    
    # mIOU
    iou = (diag + 1e-6) / (row_sum + col_sum - diag + 1e-6)
    miou = iou.mean()
    
    # mDICE
    dice = (2 * diag + 1e-6) / (row_sum + col_sum + 1e-6)
    mdice = dice.mean()
    
    return miou, mdice

if __name__ == "__main__":
    # Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 15
    BATCH_SIZE = 16 # Standard float32 uses more memory than AMP
    NUM_CLASSES = 23
    SEED = 42

    # Data Prep
    rgb_dir, mask_dir = 'data/CameraRGB', 'data/CameraMask'
    rgb_paths = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.png')])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    dataset = MyDataset(rgb_paths, mask_paths)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size], 
                                     generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = {'loss': [], 'miou': [], 'mdice': []}

    # Create directory for saving artifacts
    os.makedirs('Question2', exist_ok=True)

    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, epoch_iou, epoch_dice = 0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, masks in loop:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            
            # Forward Pass (Standard Float32)
            preds = model(imgs)
            loss = criterion(preds, masks)
            
            # Backward Pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                iou, dice = get_fast_metrics(preds, masks, NUM_CLASSES)
            
            epoch_loss += loss.item()
            epoch_iou += iou.item()
            epoch_dice += dice.item()
            loop.set_postfix(loss=loss.item())

        # Average metrics for the epoch
        avg_loss = epoch_loss / len(train_loader)
        avg_iou = epoch_iou / len(train_loader)
        avg_dice = epoch_dice / len(train_loader)

        history['loss'].append(avg_loss)
        history['miou'].append(avg_iou)
        history['mdice'].append(avg_dice)

        # PRINT METRICS AFTER EACH EPOCH
        print(f"\n[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f} | mIOU: {avg_iou:.4f} | mDICE: {avg_dice:.4f}\n")

    # --- SAVE FINAL MODEL ---
    model_path = "Question2/final_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")

    # --- FINAL TEST EVALUATION ---
    model.eval()
    test_iou, test_dice = 0, 0
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            preds = model(imgs)
            iou, dice = get_fast_metrics(preds, masks, NUM_CLASSES)
            test_iou += iou.item()
            test_dice += dice.item()
    
    print(f"\nFinal Test Results -> mIOU: {test_iou/len(test_loader):.4f}, mDICE: {test_dice/len(test_loader):.4f}")

    # Plotting
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.plot(history['loss']); plt.title('Training Loss')
    plt.subplot(1, 3, 2); plt.plot(history['miou']); plt.title('Training mIOU')
    plt.subplot(1, 3, 3); plt.plot(history['mdice']); plt.title('Training mDICE')
    plt.savefig('Question2/training_curves.png')
    plt.show()
