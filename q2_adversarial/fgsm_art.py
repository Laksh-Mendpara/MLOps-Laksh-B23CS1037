import torch
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from tqdm import tqdm
import numpy as np
import logging

def evaluate_fgsm_art(model, test_loader, config, epsilons):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=None,
        input_shape=(3, 32, 32),
        nb_classes=config.NUM_CLASSES,
        device_type=config.art_device_type(),
    )
    accuracies, examples = [], []
    
    for eps in epsilons:
        if eps == 0.0:
            correct, total = 0, 0
            for data, target in tqdm(test_loader, desc=f"FGSM ART Epsilon {eps}"):
                data_np, target_np = data.cpu().numpy(), target.cpu().numpy()
                preds = np.argmax(classifier.predict(data_np), axis=1)
                correct += np.sum(preds == target_np)
                total += len(target_np)
            acc = correct / float(total)
            logging.info(f"Epsilon: {eps}\tAccuracy = {acc:.4f}")
            accuracies.append(acc)
            examples.append([])
            continue
            
        attack = FastGradientMethod(estimator=classifier, eps=eps)
        correct, total, adv_examples = 0, 0, []
        
        for data, target in tqdm(test_loader, desc=f"FGSM ART Epsilon {eps}"):
            data_np, target_np = data.cpu().numpy(), target.cpu().numpy()
            preds_before = np.argmax(classifier.predict(data_np), axis=1)
            data_adv_np = attack.generate(x=data_np)
            preds_after = np.argmax(classifier.predict(data_adv_np), axis=1)
            
            correct += np.sum(preds_after == target_np)
            total += len(target_np)
            
            if len(adv_examples) < config.ADV_SAMPLE_LOG_COUNT:
                for i in range(len(data_np)):
                    if len(adv_examples) < config.ADV_SAMPLE_LOG_COUNT:
                        adv_examples.append((int(target_np[i]), int(preds_before[i]), int(preds_after[i]), data_np[i], data_adv_np[i]))
                        
        acc = correct / float(total)
        logging.info(f"Epsilon: {eps}\tAccuracy = {acc:.4f}")
        accuracies.append(acc)
        examples.append(adv_examples)
        
    return accuracies, examples
