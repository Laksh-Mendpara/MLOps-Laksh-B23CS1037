import os
import torch
import warnings

class Config:
    NUM_CLASSES = 10

    SEED = int(os.getenv("Q2_SEED", "42"))
    BATCH_SIZE = int(os.getenv("Q2_BATCH_SIZE", "128"))
    DETECTOR_BATCH_SIZE = int(os.getenv("Q2_DETECTOR_BATCH_SIZE", "64"))
    EPOCHS = int(os.getenv("Q2_EPOCHS", "30"))
    DETECTOR_EPOCHS = int(os.getenv("Q2_DETECTOR_EPOCHS", "8"))
    NUM_WORKERS = int(os.getenv("Q2_NUM_WORKERS", "4"))

    CLEAN_LEARNING_RATE = float(os.getenv("Q2_CLEAN_LR", "0.1"))
    CLEAN_MOMENTUM = float(os.getenv("Q2_CLEAN_MOMENTUM", "0.9"))
    CLEAN_WEIGHT_DECAY = float(os.getenv("Q2_CLEAN_WEIGHT_DECAY", "5e-4"))
    LABEL_SMOOTHING = float(os.getenv("Q2_LABEL_SMOOTHING", "0.1"))

    DETECTOR_LEARNING_RATE = float(os.getenv("Q2_DETECTOR_LR", "3e-4"))
    DETECTOR_WEIGHT_DECAY = float(os.getenv("Q2_DETECTOR_WEIGHT_DECAY", "1e-4"))
    MIN_CLEAN_TEST_ACC = float(os.getenv("Q2_MIN_CLEAN_TEST_ACC", "72.0"))
    MIN_DETECTOR_TEST_ACC = float(os.getenv("Q2_MIN_DETECTOR_TEST_ACC", "70.0"))
    DISPLAY_EPSILON = float(os.getenv("Q2_DISPLAY_EPSILON", "0.1"))
    RETRAIN_IF_CHECKPOINT_BELOW_TARGET = os.getenv("Q2_RETRAIN_IF_CHECKPOINT_BELOW_TARGET", "1") == "1"
    AMP_ENABLED = os.getenv("Q2_AMP_ENABLED", "1") == "1"
    
    GPU_INDEX = int(os.getenv("GPU_INDEX", "0"))
    DEVICE = f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu"
    
    DATA_DIR = "./data"
    OUTPUT_DIR = "./output/q2_adversarial"
    
    HF_TOKEN = os.getenv("HF_Token", "")
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT = "MLOps-Assignment-5"
    
    EPSILON_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    DETECTOR_ATTACK_EPS = 0.1
    DETECTOR_ATTACK_STEP = 0.01
    DETECTOR_ATTACK_ITERS = 20
    ADV_SAMPLE_LOG_COUNT = 10
    
    @classmethod
    def setup(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

    @classmethod
    def art_device_type(cls):
        return "gpu" if cls.DEVICE.startswith("cuda") else "cpu"

    @classmethod
    def validate_runtime(cls):
        if cls.DEVICE.startswith("cuda"):
            device_count = torch.cuda.device_count()
            if cls.GPU_INDEX >= device_count:
                warnings.warn(
                    f"Requested GPU_INDEX={cls.GPU_INDEX}, but only {device_count} CUDA device(s) are visible. Falling back to cuda:0."
                )
                cls.GPU_INDEX = 0
                cls.DEVICE = "cuda:0"
            torch.cuda.set_device(cls.GPU_INDEX)

    @classmethod
    def use_amp(cls):
        return cls.AMP_ENABLED and cls.DEVICE.startswith("cuda")
