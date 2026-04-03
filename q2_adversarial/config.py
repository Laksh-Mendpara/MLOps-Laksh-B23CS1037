import os
import torch

class Config:
    NUM_CLASSES = 10
    
    BATCH_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    DATA_DIR = "./data"
    OUTPUT_DIR = "./output/q2_adversarial"
    
    HF_TOKEN = os.getenv("HF_Token", "")
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT = "MLOps-Assignment-5"
    
    EPSILON_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    @classmethod
    def setup(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
