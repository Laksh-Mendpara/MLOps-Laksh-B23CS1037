import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod
from tqdm import tqdm
import numpy as np
import os
import wandb
import logging

def generate_adversarial_dataset(model, dataloader, config, attack_type="PGD"):
    model.eval()
    classifier = PyTorchClassifier(model=model, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(), optimizer=None, input_shape=(3, 32, 32), nb_classes=10, device_type=config.DEVICE)
    
    if attack_type == "PGD":
        attack = ProjectedGradientDescent(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=20)
    elif attack_type == "BIM":
        attack = BasicIterativeMethod(estimator=classifier, eps=0.1, eps_step=0.01, max_iter=20)
        
    all_images, all_labels = [], []
    max_batches = 20
    
    for i, (data, _) in enumerate(tqdm(dataloader, desc=f"Generating {attack_type} Dataset")):
        if i >= max_batches: break
        data_np = data.cpu().numpy()
        data_adv_np = attack.generate(x=data_np)
        
        all_images.extend([data_np, data_adv_np])
        all_labels.extend([np.zeros(len(data_np)), np.ones(len(data_adv_np))])
        
    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)
    dataset = TensorDataset(torch.tensor(all_images, dtype=torch.float32), torch.tensor(all_labels, dtype=torch.long))
    return dataset

def train_detector(detector_model, train_loader, val_loader, config, attack_type):
    detector_model = detector_model.to(config.DEVICE)
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(detector_model.parameters(), lr=1e-4)
    best_val_acc, best_model_path = 0.0, os.path.join(config.OUTPUT_DIR, f"detector_{attack_type}.pth")
    
    for epoch in range(1, 6):
        detector_model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train {attack_type} Detector]"):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = detector_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        detector_model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = detector_model(inputs)
                val_loss += criterion(outputs, labels).item()
                correct += outputs.max(1)[1].eq(labels).sum().item()
                total += labels.size(0)
                
        val_acc = 100. * correct / total
        logging.info(f"[{attack_type} Detector] Epoch {epoch} - Val Acc: {val_acc:.2f}%")
        
        if wandb.run is not None:
            wandb.log({f"{attack_type}_detector_epoch": epoch, f"{attack_type}_detector_val_acc": val_acc})
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(detector_model.state_dict(), best_model_path)
            
    return best_val_acc
