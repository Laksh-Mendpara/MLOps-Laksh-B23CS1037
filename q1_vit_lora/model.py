import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model, TaskType

def get_model(config, use_lora=True, r=8, alpha=8):
    hf_token = config.HF_TOKEN if config.HF_TOKEN else False
    model = ViTForImageClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_CLASSES,
        ignore_mismatched_sizes=True,
        token=hf_token
    )
    
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            r=r,
            lora_alpha=alpha,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            modules_to_save=["classifier"]
        )
        model = get_peft_model(model, lora_config)
    else:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    model.to(config.DEVICE)
    return model
