import itertools
import argparse
import wandb
import os
import torch
import csv
from config import Config
from dataset import get_dataloaders
from model import get_model
from train import train_model
from evaluate import evaluate_and_plot
from push_to_hub import push_model_to_hub
from logger_utils import setup_logger
import logging


def save_epoch_history(config, run_name, history):
    history_path = os.path.join(config.OUTPUT_DIR, f"{run_name}_train_val_table.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)
    return history_path


def save_results_table(config, results):
    table_path = os.path.join(config.OUTPUT_DIR, "q1_test_results_table.csv")
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "LoRA layers (with/without)",
                "Rank",
                "Alpha",
                "Dropout",
                "Overall Test Accuracy",
                "Trainable Parameters used",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    return table_path


def format_result_row(lora_label, rank, alpha, config, test_acc, model_params):
    return {
        "LoRA layers (with/without)": lora_label,
        "Rank": rank,
        "Alpha": alpha,
        "Dropout": config.LORA_DROPOUT,
        "Overall Test Accuracy": round(test_acc, 4),
        "Trainable Parameters used": model_params,
    }


def run_smoke_test():
    Config.setup()
    setup_logger(Config.OUTPUT_DIR, "q1_vit_lora_test.log")
    runtime = Config.validate_runtime()
    logging.info("Starting Q1 smoke test...")
    logging.info(
        "PyTorch runtime: torch=%s, cuda_build=%s, cuda_available=%s, device_count=%s, selected_device=%s, device_name=%s",
        runtime["torch_version"],
        runtime["torch_cuda_build"],
        runtime["cuda_available"],
        runtime["device_count"],
        runtime["selected_device"],
        runtime["device_name"],
    )

    train_loader, _, class_names = get_dataloaders(Config)
    inputs, labels = next(iter(train_loader))
    inputs = inputs[:2].to(Config.DEVICE)
    labels = labels[:2].to(Config.DEVICE)

    for model_name, kwargs in [
        ("baseline", {"use_lora": False}),
        ("lora", {"use_lora": True, "r": 2, "alpha": 2}),
        ("partial_unfreeze_lora", {"use_lora": True, "r": 2, "alpha": 2, "partial_unfreeze": True}),
    ]:
        model = get_model(Config, **kwargs)
        model.eval()
        with torch.no_grad():
            logits = model(inputs).logits
        assert logits.shape == (2, Config.NUM_CLASSES), f"{model_name} logits shape mismatch: {logits.shape}"
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info("%s smoke test passed: logits=%s trainable_params=%s", model_name, tuple(logits.shape), trainable)

    logging.info("Loaded %s CIFAR-100 classes for Q1 smoke test.", len(class_names))
    print("Q1 --test passed")


def main(test_mode=False):
    if test_mode:
        run_smoke_test()
        return

    Config.setup()
    setup_logger(Config.OUTPUT_DIR, "q1_vit_lora.log")
    runtime = Config.validate_runtime()
    logging.info("Starting Q1 Experiments...")
    logging.info(f"Active Device: {Config.DEVICE}")
    logging.info(
        "PyTorch runtime: torch=%s, cuda_build=%s, cuda_available=%s, device_count=%s, selected_device=%s, device_name=%s",
        runtime["torch_version"],
        runtime["torch_cuda_build"],
        runtime["cuda_available"],
        runtime["device_count"],
        runtime["selected_device"],
        runtime["device_name"],
    )
    
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
    model, _, history = train_model(model, train_loader, test_loader, Config, run_name, epochs=Config.EPOCHS)
    save_epoch_history(Config, run_name, history)
    test_acc, _ = evaluate_and_plot(model, test_loader, class_names, Config, run_name)
    results.append(format_result_row("without", "-", "-", Config, test_acc, model_params))
    wandb.finish()
    
    # 2. LoRA Settings
    ranks, alphas = [2, 4, 8], [2, 4, 8]
    for r, alpha in itertools.product(ranks, alphas):
        run_name = f"lora_r{r}_alpha{alpha}"
        wandb.init(project=Config.WANDB_PROJECT, name=run_name, group="lora_experiments", reinit=True, config={"r":r, "alpha":alpha})
        model = get_model(Config, use_lora=True, r=r, alpha=alpha)
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model, _, history = train_model(model, train_loader, test_loader, Config, run_name, epochs=Config.EPOCHS)
        save_epoch_history(Config, run_name, history)
        test_acc, _ = evaluate_and_plot(model, test_loader, class_names, Config, run_name)
        results.append(format_result_row("with", r, alpha, Config, test_acc, model_params))
        
        if test_acc > best_overall_acc:
            best_overall_acc = test_acc
            best_config = {"r": r, "alpha": alpha}
            model.save_pretrained(best_model_path)
        wandb.finish()

    if best_config:
        run_name = f"partial_unfreeze_lora_r{best_config['r']}_alpha{best_config['alpha']}"
        wandb.init(
            project=Config.WANDB_PROJECT,
            name=run_name,
            group="step7_partial_unfreeze",
            reinit=True,
            config={
                "r": best_config["r"],
                "alpha": best_config["alpha"],
                "partial_unfreeze_last_n": Config.PARTIAL_UNFREEZE_LAST_N,
            },
        )
        model = get_model(
            Config,
            use_lora=True,
            r=best_config["r"],
            alpha=best_config["alpha"],
            partial_unfreeze=True,
        )
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model, _, history = train_model(
            model, train_loader, test_loader, Config, run_name, epochs=Config.EPOCHS
        )
        save_epoch_history(Config, run_name, history)
        test_acc, _ = evaluate_and_plot(model, test_loader, class_names, Config, run_name)
        results.append(
            format_result_row(
                f"with (partial-unfreeze last {Config.PARTIAL_UNFREEZE_LAST_N})",
                best_config["r"],
                best_config["alpha"],
                Config,
                test_acc,
                model_params,
            )
        )
        wandb.finish()
        
    print("\n=== FINAL RESULTS TABLE ===")
    for r in results:
        print(r)

    results_path = save_results_table(Config, results)
    logging.info("Saved Q1 test results table to %s", results_path)
    
    if best_config:
        print(f"\nBest config: {best_config}")
        push_model_to_hub(best_model_path, "b23cs1037/ViT-S-LoRA-CIFAR100")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q1 ViT LoRA experiments.")
    parser.add_argument("--test", action="store_true", help="Run a lightweight smoke test for Q1.")
    args = parser.parse_args()
    main(test_mode=args.test)
