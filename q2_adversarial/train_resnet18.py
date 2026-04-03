import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import os
import logging

def train_resnet18(model, train_loader, val_loader, config):
    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.CLEAN_LEARNING_RATE,
        momentum=config.CLEAN_MOMENTUM,
        weight_decay=config.CLEAN_WEIGHT_DECAY,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    scaler = GradScaler(enabled=config.use_amp())

    best_val_acc = 0.0
    best_model_path = os.path.join(config.OUTPUT_DIR, "resnet18_clean.pth")
    best_model_state = None
    history = []

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]"):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=config.use_amp()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
                with torch.autocast(device_type="cuda", enabled=config.use_amp()):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        val_loss /= len(val_loader)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch %s - Train Loss: %.4f, Train Acc: %.2f%% | Val Loss: %.4f, Val Acc: %.2f%% | LR: %.6f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
        )
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "clean_train_loss": train_loss,
                "clean_train_acc": train_acc,
                "clean_val_loss": val_loss,
                "clean_val_acc": val_acc,
                "clean_lr": current_lr,
            })
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "state_dict": best_model_state,
                    "best_val_acc": best_val_acc,
                    "history": history,
                },
                best_model_path,
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_model_path, best_val_acc, history
