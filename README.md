# MLOps Assignment 5: ViT-S LoRA & Adversarial Attacks

This repository contains the implementation for Assignment 5, covering modular codebase for Vision Transformer finetuned with LoRA and Adversarial Attacks on ResNet models.

## Project Structure
- `q1_vit_lora/`: ViT-Small finetuning on CIFAR-100 with PEFT LoRA.
- `q2_adversarial/`: Adversarial attacks (FGSM, PGD, BIM) on ResNet18/34 (CIFAR-10).
- `output/`: Directory where all logs, weights, and plots are saved.
- `data/`: Dataset storage.

## Setup & Execution

### Option 1: Using Docker (Recommended)
This method ensures all dependencies and GPU drivers are correctly configured.

1.  **Configure Environment**:
    Create a `.env` file with your credentials:
    ```env
    WANDB_API_KEY=your_key
    HF_Token=your_token
    ```

2.  **Run the Pipeline**:
    The system will automatically run Q1 followed by Q2 sequentially.
    ```bash
    docker compose up --build -d
    ```

3.  **Monitor Logs**:
    ```bash
    # View Q1 logs live
    docker logs -f mlops-laksh-b23cs1037-q1_vit_lora-1
    ```

### Option 2: Running Locally (Without Docker)
Requires `uv` and a Python 3.10+ environment.

1.  **Install dependencies**:
    ```bash
    uv sync
    ```

2.  **Part 1: ViT LoRA (CIFAR-100)**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/q1_vit_lora
    python q1_vit_lora/run_all.py
    ```

3.  **Part 2: Adversarial Attacks (CIFAR-10)**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/q2_adversarial
    python q2_adversarial/run_all.py
    ```

## Proper Logging
- **WandB**: Visualization of all training metrics, gradients, and adversarial examples.
- **File Logs**: Every run creates a persistent log file in `output/` (e.g., `output/q1_vit_lora/q1_vit_lora.log`).
- **Clean Output**: Library noise (from `transformers` and `httpx`) is filtered out to show only relevant training and evaluation status.

## Experiments & Results
Results are automatically logged to the [WandB Dashboard](https://wandb.ai/b23cs1037-iit-jodhpur/MLOps-Assignment-5).

| Experiment | Target | Metric | Result |
|------------|--------|--------|--------|
| Q1 Baseline| Head   | Acc    | TBD    |
| Q1 LoRA    | Q,K,V  | Acc    | TBD    |
| Q2 Clean   | R18    | Acc    | TBD    |
| Q2 FGSM    | Attk   | Drop   | TBD    |
