import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import os
import logging
from config import Config
from dataset import get_dataloaders
from model import get_resnet18

def train_resnet18(model, train_loader, val_loader, config):
    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_acc = 0.0
    best_model_path = os.path.join(config.OUTPUT_DIR, "resnet18_clean.pth")
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]"):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100. * correct / total
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Val]"):
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_acc = 100. * correct / total
        val_loss /= len(val_loader)
        scheduler.step(val_acc)
        
        logging.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch, "clean_train_loss": train_loss, "clean_train_acc": train_acc,
                "clean_val_loss": val_loss, "clean_val_acc": val_acc
            })
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            
    return best_model_path, best_val_acc
