import os
import torch
import warnings

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
    PARTIAL_UNFREEZE_LAST_N = int(os.getenv("PARTIAL_UNFREEZE_LAST_N", "2"))
    
    # Device
    GPU_INDEX = int(os.getenv("GPU_INDEX", "0"))
    DEVICE = f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu"
    
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

    @classmethod
    def get_runtime_diagnostics(cls):
        diagnostics = {
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": 0,
            "selected_device": cls.DEVICE,
            "device_name": None,
        }

        if diagnostics["cuda_available"]:
            diagnostics["device_count"] = torch.cuda.device_count()
            if cls.GPU_INDEX < diagnostics["device_count"]:
                diagnostics["device_name"] = torch.cuda.get_device_name(cls.GPU_INDEX)
            else:
                diagnostics["device_name"] = torch.cuda.get_device_name(0)

        return diagnostics

    @classmethod
    def validate_runtime(cls):
        diagnostics = cls.get_runtime_diagnostics()
        if diagnostics["cuda_available"] and cls.GPU_INDEX >= diagnostics["device_count"]:
            warnings.warn(
                f"Requested GPU_INDEX={cls.GPU_INDEX}, but only {diagnostics['device_count']} CUDA device(s) are visible. "
                "Falling back to cuda:0."
            )
            cls.GPU_INDEX = 0
            cls.DEVICE = "cuda:0"
            diagnostics["selected_device"] = cls.DEVICE
            diagnostics["device_name"] = torch.cuda.get_device_name(0)
        if cls.DEVICE == "cpu" and diagnostics["torch_cuda_build"] is not None:
            warnings.warn(
                "PyTorch was installed with CUDA support, but CUDA is unavailable at runtime. "
                f"torch={diagnostics['torch_version']}, cuda_build={diagnostics['torch_cuda_build']}. "
                "This usually means the installed PyTorch CUDA build is incompatible with the host NVIDIA driver."
            )
        return diagnostics
