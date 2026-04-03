import matplotlib.pyplot as plt
import numpy as np
import wandb
import os

def denormalize(img):
    return np.clip(img.transpose(1, 2, 0), 0, 1)

def plot_adversarial_examples(examples, class_names, config, title="FGSM Adversarial Examples", filename="fgsm_examples.png"):
    if not examples: return
    num_examples = min(len(examples), 10)
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, 4 * num_examples))
    if num_examples == 1: axes = [axes]
    
    for i in range(num_examples):
        init_pred, final_pred, orig_ex, adv_ex = examples[i]
        axes[i][0].imshow(denormalize(orig_ex))
        axes[i][0].set_title(f"Original: {class_names[init_pred]}\n(Pred: {class_names[init_pred]})")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(denormalize(adv_ex))
        axes[i][1].set_title(f"Adversarial\n(Pred: {class_names[final_pred]})")
        axes[i][1].axis('off')
        
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(config.OUTPUT_DIR, filename)
    plt.savefig(plot_path)
    plt.close()
    
    if wandb.run is not None:
        wandb.log({title: wandb.Image(plot_path)})
