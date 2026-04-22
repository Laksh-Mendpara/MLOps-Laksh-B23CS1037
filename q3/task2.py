import os
import torch
import shutil
from diffusers import StableDiffusionPipeline
from optimum.onnxruntime import ORTStableDiffusionPipeline

# Paths
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LORA_WEIGHTS = "Question3/lora_weights"
FUSED_MODEL_PATH = "Question3/fused_model"
ONNX_OUTPUT_PATH = "Question3/onnx_model"

def get_dir_size_gb(directory):
    """Calculates directory size in GB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    return total_size / (1024**3)

def export_to_onnx():
    os.makedirs("Question3", exist_ok=True)

    # 1. Merge LoRA Weights
    print("Step 1: Merging LoRA weights with base model...")
    # Use float32 for export to ensure max compatibility
    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    pipe.load_lora_weights(LORA_WEIGHTS)
    
    # fuse_lora() merges the weights into the base layers for a standalone model
    pipe.fuse_lora() 
    pipe.save_pretrained(FUSED_MODEL_PATH)
    print(f"Fused model saved to {FUSED_MODEL_PATH}")

    # 2. Export to ONNX
    print("Step 2: Exporting fused model to ONNX...")
    # The class is ORTStableDiffusionPipeline in modern Optimum
    model = ORTStableDiffusionPipeline.from_pretrained(
        FUSED_MODEL_PATH, 
        export=True,
        # On V100, we stay in fp32 for the base export; optimization happens later
    )
    model.save_pretrained(ONNX_OUTPUT_PATH)
    print(f"ONNX model exported to {ONNX_OUTPUT_PATH}")

    # 3. Measurement (Task 2 Requirement)
    # The original baseline model files total roughly 5.35 GB
    original_size = 5.35 
    onnx_size = get_dir_size_gb(ONNX_OUTPUT_PATH)

    print("\n" + "="*30)
    print("TASK 2 RESULTS")
    print("="*30)
    print(f"Original Baseline Model Size: {original_size:.2f} GB")
    print(f"Combined ONNX Models Size:   {onnx_size:.2f} GB")
    print("="*30)
    
    # Cleanup fused temp folder
    if os.path.exists(FUSED_MODEL_PATH):
        shutil.rmtree(FUSED_MODEL_PATH)

if __name__ == "__main__":
    export_to_onnx()