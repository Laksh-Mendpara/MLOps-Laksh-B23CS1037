import itertools
import wandb
import os
import torch
from config import Config
from dataset import get_dataloaders
from model import get_model
from train import train_model
from evaluate import evaluate_and_plot
from push_to_hub import push_model_to_hub
from logger_utils import setup_logger
import logging

def main():
    Config.setup()
    setup_logger(Config.OUTPUT_DIR, "q1_vit_lora.log")
    logging.info("Starting Q1 Experiments...")
    logging.info(f"Active Device: {Config.DEVICE}")
    
    if Config.WANDB_API_KEY:
        wandb.login(key=Config.WANDB_API_KEY)
    
    train_loader, test_loader, class_names = get_dataloaders(Config)
    results = []
    best_overall_acc = 0
    best_config = None
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    
    # 1. Baseline
    run_name = "baseline_no_lora"
    wandb.init(project=Config.WANDB_PROJECT, name=run_name, group="baseline", reinit=True)
    model = get_model(Config, use_lora=False)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model, _ = train_model(model, train_loader, test_loader, Config, run_name, epochs=Config.EPOCHS)
    test_acc, _ = evaluate_and_plot(model, test_loader, class_names, Config, run_name)
    results.append({"LoRA": "without", "Rank": "-", "Alpha": "-", "Acc": test_acc, "Params": model_params})
    wandb.finish()
    
    # 2. LoRA Settings
    ranks, alphas = [2, 4, 8], [2, 4, 8]
    for r, alpha in itertools.product(ranks, alphas):
        run_name = f"lora_r{r}_alpha{alpha}"
        wandb.init(project=Config.WANDB_PROJECT, name=run_name, group="lora_experiments", reinit=True, config={"r":r, "alpha":alpha})
        model = get_model(Config, use_lora=True, r=r, alpha=alpha)
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model, _ = train_model(model, train_loader, test_loader, Config, run_name, epochs=Config.EPOCHS)
        test_acc, _ = evaluate_and_plot(model, test_loader, class_names, Config, run_name)
        results.append({"LoRA": "with", "Rank": r, "Alpha": alpha, "Acc": test_acc, "Params": model_params})
        
        if test_acc > best_overall_acc:
            best_overall_acc = test_acc
            best_config = {"r": r, "alpha": alpha}
            model.save_pretrained(best_model_path)
        wandb.finish()
        
    print("\n=== FINAL RESULTS TABLE ===")
    for r in results:
        print(r)
    
    if best_config:
        print(f"\nBest config: {best_config}")
        push_model_to_hub(best_model_path, "b23cs1037/ViT-S-LoRA-CIFAR100")

if __name__ == "__main__":
    main()
