import wandb
import torch
import os
import matplotlib.pyplot as plt
from config import Config
from dataset import get_dataloaders
from model import get_resnet18, get_resnet34
from train_resnet18 import train_resnet18
from fgsm_scratch import evaluate_fgsm_scratch
from fgsm_art import evaluate_fgsm_art
from adversarial_detector import generate_adversarial_dataset, train_detector
from visualize import plot_adversarial_examples
from torch.utils.data import DataLoader, random_split
from logger_utils import setup_logger
import logging

def plot_accuracy_drop(epsilons, acc_scratch, acc_art, config):
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, [a * 100 for a in acc_scratch], 'o-', label='FGSM (Scratch)')
    plt.plot(epsilons, [a * 100 for a in acc_art], 's-', label='FGSM (IBM ART)')
    plt.title("Accuracy vs Perturbation Strength (Epsilon)")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(config.OUTPUT_DIR, "accuracy_drop.png")
    plt.savefig(plot_path)
    plt.close()
    if wandb.run is not None:
        wandb.log({"Accuracy vs Epsilon Drop": wandb.Image(plot_path)})

def main():
    Config.setup()
    setup_logger(Config.OUTPUT_DIR, "q2_adversarial.log")
    logging.info("Starting Q2 Adversarial Attack Experiments...")
    logging.info(f"Active Device: {Config.DEVICE}")
    
    if Config.WANDB_API_KEY:
        wandb.login(key=Config.WANDB_API_KEY)
        
    wandb.init(project=Config.WANDB_PROJECT, name="Q2_Adversarial_Attacks", group="Q2")
    train_loader, test_loader, class_names = get_dataloaders(Config)
    
    print("\n=== Training ResNet18 on Clean CIFAR-10 ===")
    model = get_resnet18(Config.NUM_CLASSES)
    model_path = os.path.join(Config.OUTPUT_DIR, "resnet18_clean.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(Config.DEVICE)
    else:
        train_resnet18(model, train_loader, test_loader, Config)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        
    print("\n=== Running FGSM Attack (Scratch) ===")
    acc_scratch, examples_scratch = evaluate_fgsm_scratch(model, test_loader, Config, Config.EPSILON_VALUES)
    target_idx = min(3, len(examples_scratch)-1)
    if examples_scratch[target_idx]:
        plot_adversarial_examples(examples_scratch[target_idx], class_names, Config, f"FGSM Scratch (eps={Config.EPSILON_VALUES[target_idx]})", "fgsm_scratch_samples.png")
    
    print("\n=== Running FGSM Attack (IBM ART) ===")
    acc_art, examples_art = evaluate_fgsm_art(model, test_loader, Config, Config.EPSILON_VALUES)
    if examples_art[target_idx]:
        plot_adversarial_examples(examples_art[target_idx], class_names, Config, f"FGSM ART (eps={Config.EPSILON_VALUES[target_idx]})", "fgsm_art_samples.png")
                                  
    plot_accuracy_drop(Config.EPSILON_VALUES, acc_scratch, acc_art, Config)
    
    for attack in ["PGD", "BIM"]:
        print(f"\n=== Training {attack} Adversarial Detector ===")
        dataset = generate_adversarial_dataset(model, test_loader, Config, attack_type=attack)
        train_size = int(0.8 * len(dataset))
        train_d, val_d = random_split(dataset, [train_size, len(dataset) - train_size])
        train_l = DataLoader(train_d, batch_size=64, shuffle=True)
        val_l = DataLoader(val_d, batch_size=64, shuffle=False)
        det_acc = train_detector(get_resnet34(num_classes=2), train_l, val_l, Config, attack)
        wandb.log({f"Final_{attack}_Detection_Acc": det_acc})
        
    wandb.finish()

if __name__ == "__main__":
    main()
