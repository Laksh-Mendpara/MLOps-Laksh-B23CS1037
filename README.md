# MLOps Assignment 3 – Fine-Tuning & Docker Deployment

**Student:** Laksh Mendpara (B23CS1037)

---

## Overview

This repository implements an end-to-end ML workflow:

1. **Fine-tune** `distilbert-base-cased` on GoodReads book-review data to classify text into **8 genres**.
2. **Dockerize** both the training and production-evaluation pipelines.
3. **Publish** the trained model to the Hugging Face Hub.
4. **Re-evaluate** by pulling the model directly from the Hub to verify reproducibility.

---

## Repository Structure

```
MLOps-Laksh-B23CS1037/
├── ML_DL_Ops_Ass_3_Fine_Tuning_Classification.ipynb  # Original notebook (Task 1)
├── src/
│   ├── data.py        # Data loading & train/test split (Task 3, 5)
│   ├── utils.py       # MyDataset + compute_metrics (Task 3)
│   ├── train.py       # Training script with Trainer API (Task 5)
│   └── evaluate.py    # Local & Hub evaluation (Task 6, 8)
├── Dockerfile         # Dev / training image (Task 2)
├── Dockerfile.eval    # Production evaluation image (Task 9)
├── entrypoint.sh      # Auto-runs evaluate.py on container start
├── requirements.txt   # Python dependencies
├── results/           # Saved evaluation JSON files (generated at runtime)
├── SHORT_REPORT.md    # Model selection & summary report (Task 10)
└── README.md          # This file
```

---

## Model

| Item | Value |
|------|-------|
| Base model | `distilbert-base-cased` |
| Task | Multi-class text classification (8 genres) |
| Dataset | GoodReads by-genre reviews (UCSD) |
| Hugging Face Hub | [laksh-B23CS1037/distilbert-book-genre](https://huggingface.co/laksh-B23CS1037/distilbert-book-genre) |

---

## Quick Start

### Training - Training Image (Task 2)

```bash
# Build
docker build --network=host -t genre-train .

# Run interactively (mount results for persistence)
docker run --rm -it \
    --gpus all --shm-size=8g \
    --network=host \
    -v $(pwd):/app \
    genre-train

# Inside the container
python src/train.py --epochs 5
```

---

### 3. Docker – Production Evaluation Image (Task 9)

This image pulls the model from the Hugging Face Hub at runtime and runs evaluation automatically.

```bash
# Build
docker build -f Dockerfile.eval --network=host -t genre-eval .

# Run (model is fetched from HF Hub; results appear in ./results/)
docker run --rm \
    -e HF_REPO=laksh-B23CS1037/distilbert-book-genre \
    --gpus all --shm-size=8g \
    --network=host \
    -v $(pwd):/app \
    genre-eval
```

---

## Evaluation Results

Results are written to `results/` as JSON files after training / evaluation runs.

| Metric | Local model | From HF Hub |
|--------|-------------|-------------|
| Accuracy | _see `results/eval_local.json`_ | _see `results/eval_hub.json`_ |

Both files should contain identical (or near-identical) metrics because the Hub model is the uploaded checkpoint of the local fine-tuned model.

---

## Links

- **Hugging Face Model:** https://huggingface.co/laksh-B23CS1037/distilbert-book-genre
- **GitHub Repository:** https://github.com/Laksh-Mendpara/MLOps-Laksh-B23CS1037
