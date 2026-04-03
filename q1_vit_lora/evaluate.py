import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def evaluate_and_plot(model, test_loader, class_names, config, run_name):
    model.eval()
    correct, total = 0, 0
    num_classes = len(class_names)
    class_correct = list(0. for _ in range(num_classes))
    class_total = list(0. for _ in range(num_classes))
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs).logits
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    overall_acc = 100. * correct / total
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%")
    
    class_acc = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(num_classes)]
        
    plt.figure(figsize=(20, 8))
    plt.bar(range(num_classes), class_acc)
    plt.xlabel('Class Index', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title(f'Class-wise Test Accuracy ({run_name})', fontsize=16)
    plt.xticks(range(0, num_classes, 5))
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(config.OUTPUT_DIR, f"{run_name}_class_acc.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    if wandb.run is not None:
        wandb.log({
            "overall_test_acc": overall_acc,
            "class_wise_accuracy_histogram": wandb.Image(plot_path)
        })
    return overall_acc, class_acc
