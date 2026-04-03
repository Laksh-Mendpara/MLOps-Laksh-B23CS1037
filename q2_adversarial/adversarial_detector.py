import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
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
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=nn.CrossEntropyLoss(),
        optimizer=None,
        input_shape=(3, 32, 32),
        nb_classes=10,
        device_type=config.art_device_type(),
    )
    
    if attack_type == "PGD":
        attack = ProjectedGradientDescent(
            estimator=classifier,
            eps=config.DETECTOR_ATTACK_EPS,
            eps_step=config.DETECTOR_ATTACK_STEP,
            max_iter=config.DETECTOR_ATTACK_ITERS,
        )
    elif attack_type == "BIM":
        attack = BasicIterativeMethod(
            estimator=classifier,
            eps=config.DETECTOR_ATTACK_EPS,
            eps_step=config.DETECTOR_ATTACK_STEP,
            max_iter=config.DETECTOR_ATTACK_ITERS,
        )
    else:
        raise ValueError(f"Unsupported attack_type: {attack_type}")
        
    all_images, all_labels = [], []
    all_examples = []
    
    for i, (data, target) in enumerate(tqdm(dataloader, desc=f"Generating {attack_type} Dataset")):
        data_np = data.cpu().numpy()
        target_np = target.cpu().numpy()
        preds_before = np.argmax(classifier.predict(data_np), axis=1)
        data_adv_np = attack.generate(x=data_np)
        preds_after = np.argmax(classifier.predict(data_adv_np), axis=1)
        
        all_images.extend([data_np, data_adv_np])
        all_labels.extend([np.zeros(len(data_np)), np.ones(len(data_adv_np))])
        if len(all_examples) < config.ADV_SAMPLE_LOG_COUNT:
            for true_label, clean_pred, adv_pred, clean_img, adv_img in zip(
                target_np, preds_before, preds_after, data_np, data_adv_np
            ):
                if len(all_examples) >= config.ADV_SAMPLE_LOG_COUNT:
                    break
                all_examples.append((int(true_label), int(clean_pred), int(adv_pred), clean_img, adv_img))
        
    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)
    dataset = TensorDataset(torch.tensor(all_images, dtype=torch.float32), torch.tensor(all_labels, dtype=torch.long))
    return dataset, all_examples

def train_detector(detector_model, train_loader, val_loader, config, attack_type):
    detector_model = detector_model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        detector_model.parameters(),
        lr=config.DETECTOR_LEARNING_RATE,
        weight_decay=config.DETECTOR_WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.DETECTOR_EPOCHS)
    scaler = GradScaler(enabled=config.use_amp())
    best_val_acc, best_model_path = 0.0, os.path.join(config.OUTPUT_DIR, f"detector_{attack_type}.pth")
    best_model_state = None
    history = []
    
    for epoch in range(1, config.DETECTOR_EPOCHS + 1):
        detector_model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train {attack_type} Detector]"):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=config.use_amp()):
                outputs = detector_model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        detector_model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                with torch.autocast(device_type="cuda", enabled=config.use_amp()):
                    outputs = detector_model(inputs)
                    val_loss += criterion(outputs, labels).item()
                val_correct += outputs.max(1)[1].eq(labels).sum().item()
                val_total += labels.size(0)
                
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0.0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "[%s Detector] Epoch %s - Train Acc: %.2f%% | Val Acc: %.2f%% | LR: %.6f",
            attack_type,
            epoch,
            train_acc,
            val_acc,
            current_lr,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / max(len(train_loader), 1),
            "train_acc": train_acc,
            "val_loss": val_loss / max(len(val_loader), 1),
            "val_acc": val_acc,
            "lr": current_lr,
        })
        
        if wandb.run is not None:
            wandb.log({
                f"{attack_type}_detector_epoch": epoch,
                f"{attack_type}_detector_train_acc": train_acc,
                f"{attack_type}_detector_val_acc": val_acc,
                f"{attack_type}_detector_train_loss": train_loss / max(len(train_loader), 1),
                f"{attack_type}_detector_val_loss": val_loss / max(len(val_loader), 1),
                f"{attack_type}_detector_lr": current_lr,
            })
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu() for k, v in detector_model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best_model_state,
                    "best_val_acc": best_val_acc,
                    "history": history,
                    "attack_type": attack_type,
                },
                best_model_path,
            )
            
    if best_model_state is not None:
        detector_model.load_state_dict(best_model_state)

    return detector_model, best_val_acc, history


def evaluate_detector(detector_model, test_loader, config):
    detector_model = detector_model.to(config.DEVICE)
    detector_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = detector_model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total if total > 0 else 0.0
