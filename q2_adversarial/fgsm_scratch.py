import torch
import torch.nn as nn
from tqdm import tqdm
import logging

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def evaluate_fgsm_scratch(model, test_loader, config, epsilons):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    accuracies, examples = [], []
    
    for eps in epsilons:
        correct, total, adv_examples = 0, 0, []
        for data, target in tqdm(test_loader, desc=f"FGSM Scratch Epsilon {eps}"):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            data.requires_grad = True
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward()
            
            perturbed_data = fgsm_attack(data, eps, data.grad.data)
            final_pred = model(perturbed_data).max(1, keepdim=True)[1]
            
            correct += (final_pred.flatten() == target).sum().item()
            total += target.size(0)
            
            if len(adv_examples) < 10 and eps > 0:
                successful_indices = (init_pred.flatten() == target) & (final_pred.flatten() != target)
                for i in range(len(data)):
                    if successful_indices[i] and len(adv_examples) < 10:
                        adv_examples.append((init_pred[i].item(), final_pred[i].item(), data[i].squeeze().detach().cpu().numpy(), perturbed_data[i].squeeze().detach().cpu().numpy()))
                        
        acc = correct / float(total)
        logging.info(f"Epsilon: {eps}\tTest Accuracy = {acc:.4f}")
        accuracies.append(acc)
        examples.append(adv_examples)
        
    return accuracies, examples
