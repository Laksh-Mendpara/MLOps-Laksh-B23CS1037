# Assignment 4 — Optimizing Transformer Translation with Ray Tune & Optuna

**Name:** Laksh Mendpara  
**Roll No.:** B23CS1037  
**Course:** MLOps — Spring 2026

---

## Links

| Resource               | URL                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| **GitHub Repository**  | [Assignment-4 branch](https://github.com/Laksh-Mendpara/MLOps-Laksh-B23CS1037/tree/Assignment-4) |
| **Hugging Face Model** | [MLOPS-Assignment-4](https://huggingface.co/Laksh-Mendpara/MLOPS-Assignment-4/tree/main)         |

---

## Overview

This project optimizes a custom **Transformer** model for English-to-Hindi translation. The baseline notebook (`baseline.ipynb`) trains a standard 6-layer Transformer for 100 epochs. We refactor the training pipeline into a modular codebase and use **Ray Tune** with the **Optuna** search algorithm and **ASHA** scheduler to find a hyperparameter configuration that beats the baseline BLEU score in significantly fewer epochs.

---

## Baseline vs. Optimized Results

| Metric                  | Baseline (100 epochs) | Optimized (20 epochs) |
| ----------------------- | :-------------------: | :-------------------: |
| **Epochs**              |          100          |        **20**         |
| **Training Time**       |        ~70 min        |      **~24 min**      |
| **Final Training Loss** |      **0.0976**       |        0.8760         |
| **BLEU Score (NLTK)**   |         75.66         |       **90.38**       |

> **Key takeaway:** The optimized model achieves a **+14.72 point BLEU improvement** in only **20% of the epochs** and **~34% of the wall-clock time**.

### Why is the loss higher but BLEU is better?

The baseline trains for 100 epochs on a small corpus (~13k pairs) which leads to overfitting — the model memorizes training tokens but doesn't generalize well. The optimized model uses stronger regularization (higher dropout, smaller batch size) and trains for fewer epochs, resulting in higher training loss but significantly better translation quality.

---

## Hyperparameters Tuned

| Hyperparameter                | Type             | Range / Choices | Optimal Value |
| ----------------------------- | ---------------- | --------------- | :-----------: |
| Learning Rate (`lr`)          | Continuous (log) | [1e-5, 1e-3]    |    1.49e-4    |
| Batch Size (`batch_size`)     | Categorical      | {16, 32, 64}    |      16       |
| Attention Heads (`num_heads`) | Categorical      | {4, 8}          |       4       |
| Feed-Forward Dim (`d_ff`)     | Categorical      | {1024, 2048}    |     1024      |
| Dropout (`dropout`)           | Continuous       | [0.1, 0.4]      |     0.335     |

- **20 trials** were run using Optuna with the ASHA scheduler (grace period = 3 epochs, reduction factor = 3).
- Trials ran concurrently across **4 GPUs** (one trial per GPU).

---

## Project Structure

```
├── data/
│   └── English-Hindi.tsv           # Raw parallel corpus (~13k pairs)
├── dataset/
│   ├── data_loader.py              # PyTorch Dataset and DataLoader
│   └── vocab.py                    # Vocabulary building and encoding
├── models/
│   └── transformer.py              # Custom Transformer architecture
├── core/
│   ├── evaluate.py                 # Inference and BLEU evaluation
│   └── tune.py                     # Ray Tune training loop
├── main.py                         # Tuning entrypoint
├── baseline.ipynb                  # Baseline training notebook (100 epochs)
├── b23cs1037_ass_4_best_model.pth  # Best model weights
├── b23cs1037_ass_4_report.tex      # LaTeX report
├── rollno_ass_4_report.txt         # Best config summary
└── output.log                      # Ray Tune execution log
```

---

## Setup & Reproduce

### Prerequisites

This project uses `uv` for dependency management.

```bash
uv sync
source .venv/bin/activate
```

Create a `.env` file with your Hugging Face token:

```
hf_token=your_hugging_face_token
```

### Run Hyperparameter Tuning

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py
```

### Quick Sanity Check (4 trials, 1 epoch each)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main.py --test-run
```

### Outputs

1. Prints the best configuration and metrics to the console.
2. Saves the best model to `b23cs1037_ass_4_best_model.pth`.
3. Creates `rollno_ass_4_report.txt` with the optimal config.
4. Pushes the model to Hugging Face automatically.

---

## Report

The detailed report is available at [`b23cs1037_ass_4_report.pdf`](b23cs1037_ass_4_report.pdf). It covers:

- Baseline metrics and methodology
- Hyperparameter search space design
- Best configuration analysis
- Comparative results and conclusions
