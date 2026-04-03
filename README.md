# MLOps Assignment 5: ViT-S LoRA & Adversarial Attacks

This repository contains the implementation for Assignment 5:
- `Q1`: ViT-S / DeiT-S finetuning on CIFAR-100 with baseline head-only training, LoRA sweeps, and optional partial-unfreeze LoRA.
- `Q2`: Adversarial attacks and adversarial detectors on CIFAR-10 using ResNet18 and ResNet34.

## Project Structure
- `q1_vit_lora/`: ViT-Small finetuning on CIFAR-100 with PEFT LoRA.
- `q2_adversarial/`: Adversarial attacks (FGSM, PGD, BIM) on ResNet18/34 (CIFAR-10).
- `output/`: Directory where all logs, weights, and plots are saved.
- `data/`: Dataset storage.

## Setup & Execution

### Option 1: Using Docker
This keeps the environment isolated and matches the assignment requirement to use Docker.

1.  **Configure Environment**:
    Create a `.env` file with your credentials:
    ```env
    WANDB_API_KEY=your_key
    HF_Token=your_token
    ```

2.  **Run the pipeline**:
    ```bash
    docker compose up --build -d
    ```

3.  **Monitor Logs**:
    ```bash
    # View Q1 logs live
    docker logs -f mlops-laksh-b23cs1037-q1_vit_lora-1
    ```

### Option 2: Running Locally
Requires `uv` and Python 3.10+.

1.  **Install dependencies**:
    ```bash
    uv sync
    ```

2.  **Set GPU if needed**:
    `GPU_INDEX` selects the CUDA device. Example: `GPU_INDEX=3`

3.  **Q1: ViT LoRA (full run)**:
    ```bash
    GPU_INDEX=3 python q1_vit_lora/run_all.py
    ```

4.  **Q1 smoke test**:
    ```bash
    GPU_INDEX=3 python q1_vit_lora/run_all.py --test
    ```

5.  **Q2: Adversarial attacks + detectors (full run)**:
    ```bash
    GPU_INDEX=3 python q2_adversarial/run_all.py
    ```

6.  **Q2 smoke test**:
    ```bash
    GPU_INDEX=3 python q2_adversarial/run_all.py --test
    ```

If you prefer `uv run`, these work too:
```bash
GPU_INDEX=3 uv run python q1_vit_lora/run_all.py
GPU_INDEX=3 uv run python q2_adversarial/run_all.py
```

## Implemented Changes

### Q1
- Baseline head-only finetuning on CIFAR-100.
- LoRA sweep across `rank in {2,4,8}` and `alpha in {2,4,8}` with dropout `0.1`.
- Optional step-7 experiment: partial-unfreeze LoRA using the best step-2 hyperparameters.
- GPU selection through `GPU_INDEX`.
- `--test` smoke-test mode for quick verification.
- Per-run train/val CSV tables and a consolidated Q1 test-results CSV.

### Q2
- Clean ResNet18 training on CIFAR-10 with automatic checkpoint validation against the assignment threshold.
- FGSM from scratch and FGSM with IBM ART.
- PGD and BIM adversarial detector training using ResNet34.
- Detector datasets are generated separately for train, val, and test splits.
- Qualitative adversarial sample plots for FGSM, PGD, and BIM with 10 clean/adversarial pairs logged to WandB.
- GPU selection through `GPU_INDEX`.
- `--test` smoke-test mode for quick verification.
- CSV summaries plus train/val curve plots for the clean classifier and both detectors.
- Saved Q2 weights: clean ResNet18, PGD detector, and BIM detector.

## Proper Logging
- **WandB**: Training metrics, attack comparisons, detector metrics, and qualitative adversarial examples.
- **File Logs**: Persistent logs are written under `output/`.
- **Smoke tests**: `--test` checks that the key pipeline pieces run correctly before a full experiment.

## Output Files

### Q1
- `output/q1_vit_lora/q1_vit_lora.log`
- `output/q1_vit_lora/*_train_val_table.csv`
- `output/q1_vit_lora/q1_test_results_table.csv`
- `output/q1_vit_lora/*_class_acc.png`

### Q2
- `output/q2_adversarial/q2_adversarial.log`
- `output/q2_adversarial/clean_resnet18_train_val_table.csv`
- `output/q2_adversarial/clean_resnet18_train_val_curves.png`
- `output/q2_adversarial/fgsm_accuracy_comparison.csv`
- `output/q2_adversarial/detector_accuracy_comparison.csv`
- `output/q2_adversarial/*_detector_train_val_table.csv`
- `output/q2_adversarial/*_detector_train_val_curves.png`
- `output/q2_adversarial/resnet18_clean.pth`
- `output/q2_adversarial/detector_PGD.pth`
- `output/q2_adversarial/detector_BIM.pth`
- `output/q2_adversarial/*.png`

## Experiments & Results
Results are automatically logged to the [WandB Dashboard](https://wandb.ai/b23cs1037-iit-jodhpur/MLOps-Assignment-5).

| Experiment | Target | Metric | Result |
|------------|--------|--------|--------|
| Q1 Baseline| Head   | Acc    | TBD    |
| Q1 LoRA    | Q,K,V  | Acc    | TBD    |
| Q2 Clean   | R18    | Acc    | TBD    |
| Q2 FGSM    | Attk   | Drop   | TBD    |
