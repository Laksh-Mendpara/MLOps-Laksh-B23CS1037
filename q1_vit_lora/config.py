import os
import torch

class Config:
    # Model - 'google/vit-small' doesn't exist, using official 'facebook/deit-small-patch16-224' (ViT-S architecture)
    MODEL_NAME = "facebook/deit-small-patch16-224"
    NUM_CLASSES = 100 # CIFAR-100
    
    # Training
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01
    
    # Fixed LoRA params
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["query", "key", "value"] 
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    DATA_DIR = "./data"
    OUTPUT_DIR = "./output/q1_vit_lora"
    
    # HuggingFace & WandB
    # Only use token if it's definitely provided and not a placeholder
    raw_token = os.getenv("HF_Token", "")
    HF_TOKEN = raw_token if raw_token and not raw_token.startswith("<") else None
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT = "MLOps-Assignment-5"
    
    @classmethod
    def setup(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
