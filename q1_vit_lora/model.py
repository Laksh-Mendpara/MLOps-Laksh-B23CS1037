import torch
import torch.nn as nn
from transformers import ViTForImageClassification

def freeze_all_but_classifier(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_last_encoder_blocks(model, num_blocks):
    encoder_layers = model.vit.encoder.layer
    total_blocks = len(encoder_layers)
    start_idx = max(total_blocks - num_blocks, 0)

    for idx in range(start_idx, total_blocks):
        for param in encoder_layers[idx].parameters():
            param.requires_grad = True

    for param in model.vit.layernorm.parameters():
        param.requires_grad = True

    return start_idx, total_blocks


def get_model(config, use_lora=True, r=8, alpha=8, partial_unfreeze=False):
    hf_token = config.HF_TOKEN if config.HF_TOKEN else False
    model = ViTForImageClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_CLASSES,
        ignore_mismatched_sizes=True,
        token=hf_token
    )

    freeze_all_but_classifier(model)

    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_kwargs = {}
        if partial_unfreeze:
            frozen_until, total_blocks = unfreeze_last_encoder_blocks(
                model, config.PARTIAL_UNFREEZE_LAST_N
            )
            lora_kwargs["layers_to_transform"] = list(range(frozen_until))
            lora_kwargs["layers_pattern"] = "layer"

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            modules_to_save=["classifier"],
            **lora_kwargs,
        )
        model = get_peft_model(model, lora_config)

    model.to(config.DEVICE)
    return model
