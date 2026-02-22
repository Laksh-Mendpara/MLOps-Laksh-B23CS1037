"""
evaluate.py  –  Evaluate a fine-tuned genre-classification model.

Modes
-----
  local    Load the model from a local directory (default).
  hub      Load the model from a Hugging Face Hub repository.
  both     Evaluate both and print a comparison table.

Outputs (all saved in results/)
--------------------------------
  eval_<tag>.json          – metrics dict (accuracy, loss, …)
  confusion_matrix_<tag>.png
  f1_per_class_<tag>.png
  misclassification_heatmap_<tag>.png
  comparison.json          – (mode=both only)

Usage examples
--------------
  python src/evaluate.py --mode local --model_path ./fine-tuned-genre-model
  python src/evaluate.py --mode hub   --hf_repo Laksh-Mendpara/MLOps-Assignment-3
  python src/evaluate.py --mode both  --model_path ./fine-tuned-genre-model \\
                                       --hf_repo Laksh-Mendpara/MLOps-Assignment-3
"""

import os
import json
import argparse
import logging
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')       # non-interactive backend; works inside Docker
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from huggingface_hub import HfApi, login
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from data import load_all_genres, make_train_test_split
from utils import MyDataset, compute_metrics
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = './results'
REVIEWS_CACHE_PATH = './genre_reviews_dict.pickle'
MAX_LENGTH = 512


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a fine-tuned genre classifier')
    parser.add_argument('--mode', choices=['local', 'hub', 'both'], default='local')
    parser.add_argument('--model_path', default='./fine-tuned-genre-model',
                        help='Local model directory')
    parser.add_argument('--hf_repo', default='Laksh-Mendpara/MLOps-Assignment-3',
                        help='HuggingFace Hub repo id')
    parser.add_argument('--sample_size', type=int, default=2000)
    parser.add_argument('--per_genre', type=int, default=1000)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_test_dataset(sample_size: int, per_genre: int):
    """Return (test_texts, test_labels, label2id, id2label)."""
    logger.info("Loading review data …")
    genre_reviews_dict = load_all_genres(sample_size=sample_size, cache_path=REVIEWS_CACHE_PATH)

    _, train_labels_all, test_texts, test_labels = make_train_test_split(
        genre_reviews_dict, per_genre=per_genre
    )

    unique_labels = sorted(set(train_labels_all))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label  = {idx: label for label, idx in label2id.items()}

    return test_texts, test_labels, label2id, id2label


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _short(label: str) -> str:
    """Abbreviated genre name so axis ticks are readable."""
    abbr = {
        'mystery_thriller_crime': 'mystery',
        'fantasy_paranormal': 'fantasy',
        'history_biography': 'history',
        'comics_graphic': 'comics',
        'young_adult': 'YA',
    }
    return abbr.get(label, label)


def plot_confusion_matrix(true_labels, pred_labels, class_names, tag: str, out_dir: str):
    """Full confusion matrix heatmap (counts)."""
    cm     = confusion_matrix(true_labels, pred_labels, labels=class_names)
    short  = [_short(c) for c in class_names]
    df_cm  = pd.DataFrame(cm, index=short, columns=short)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5,
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True',      fontsize=12)
    ax.set_title(f'Confusion Matrix — {tag}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(out_dir, f'confusion_matrix_{tag}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("[%s] Saved %s", tag, path)


def plot_f1_per_class(true_labels, pred_labels, class_names, tag: str, out_dir: str):
    """Horizontal bar chart of per-class F1 scores."""
    f1s   = f1_score(true_labels, pred_labels, labels=class_names, average=None)
    short = [_short(c) for c in class_names]

    colours = ['#2ecc71' if f >= 0.75 else '#e67e22' if f >= 0.60 else '#e74c3c' for f in f1s]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(short, f1s, color=colours, edgecolor='white', height=0.6)
    ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title(f'Per-Class F1 — {tag}', fontsize=14, fontweight='bold')
    ax.axvline(x=np.mean(f1s), linestyle='--', color='steelblue', linewidth=1.2,
               label=f'Mean F1 = {np.mean(f1s):.2f}')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    sns.despine(left=True, bottom=False)
    plt.tight_layout()

    path = os.path.join(out_dir, f'f1_per_class_{tag}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("[%s] Saved %s", tag, path)


def plot_misclassification_heatmap(true_labels, pred_labels, class_names, tag: str, out_dir: str):
    """Heatmap of misclassifications only (diagonal removed)."""
    cm    = confusion_matrix(true_labels, pred_labels, labels=class_names).astype(float)
    np.fill_diagonal(cm, 0)          # zero out correct predictions
    short = [_short(c) for c in class_names]
    df_cm = pd.DataFrame(cm, index=short, columns=short)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='.0f', cmap='Purples', linewidths=0.5,
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_xlabel('Predicted as', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title(f'Misclassification Heatmap (no diagonal) — {tag}', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(out_dir, f'misclassification_heatmap_{tag}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("[%s] Saved %s", tag, path)


def plot_accuracy_comparison(results: dict, out_dir: str):
    """Side-by-side accuracy bar for local vs hub (mode=both only)."""
    tags   = list(results.keys())
    accs   = [results[t]['accuracy'] for t in tags]
    colors = ['#3498db', '#e74c3c']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(tags, accs, color=colors[:len(tags)], width=0.4, edgecolor='white')
    ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Local vs Hub Model Accuracy', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    sns.despine()
    plt.tight_layout()

    path = os.path.join(out_dir, 'accuracy_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def generate_plots(true_labels, predicted_labels, class_names, tag: str, out_dir: str):
    """Generate all three per-model plots."""
    os.makedirs(out_dir, exist_ok=True)
    plot_confusion_matrix(true_labels, predicted_labels, class_names, tag, out_dir)
    plot_f1_per_class(true_labels, predicted_labels, class_names, tag, out_dir)
    plot_misclassification_heatmap(true_labels, predicted_labels, class_names, tag, out_dir)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_source: str,
    test_texts: list,
    test_labels: list,
    label2id: dict,
    id2label: dict,
    tag: str = 'model',
) -> dict:
    """Load model, run evaluation, save metrics + plots. Returns metrics dict."""

    logger.info("[%s] Loading tokenizer and model from: %s", tag, model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    model     = AutoModelForSequenceClassification.from_pretrained(model_source).to(device)

    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_labels_encoded = [label2id[y] for y in test_labels]
    test_dataset = MyDataset(test_encodings, test_labels_encoded)

    eval_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, 'tmp_eval'),
        per_device_eval_batch_size=16,
        report_to=[],
        # no_cuda=(device == 'cpu'),
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("[%s] Running evaluation …", tag)
    eval_metrics = trainer.evaluate()
    logger.info("[%s] Eval metrics: %s", tag, eval_metrics)

    pred_output = trainer.predict(test_dataset)
    pred_ids = pred_output.predictions.argmax(-1).flatten().tolist()
    predicted_labels = [id2label[i] for i in pred_ids]

    # Classification report (console)
    print(f"\n{'='*60}")
    print(f"Classification Report — {tag}")
    print('='*60)
    print(classification_report(test_labels, predicted_labels))

    acc = accuracy_score(test_labels, predicted_labels)
    metrics = {
        'source': model_source,
        'tag': tag,
        'accuracy': acc,
        **{k: v for k, v in eval_metrics.items()},
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f'eval_{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("[%s] Metrics saved to %s", tag, out_path)

    # Plots
    class_names = sorted(label2id.keys())
    generate_plots(test_labels, predicted_labels, class_names, tag, RESULTS_DIR)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    HF_TOKEN = os.environ.get('HF_TOKEN', '').strip()
    if not HF_TOKEN:
        raise ValueError("Please set the HF_TOKEN environment variable")
    try:
        login(HF_TOKEN)
        _hf_username = HfApi().whoami()
    except Exception as e:
        raise ValueError("Invalid HF_TOKEN") from e
    logger.info("Using HF user: %s", _hf_username['name'])

    test_texts, test_labels, label2id, id2label = build_test_dataset(
        args.sample_size, args.per_genre
    )

    results = {}

    if args.mode in ('local', 'both'):
        results['local'] = evaluate_model(
            args.model_path,
            test_texts, test_labels,
            label2id, id2label,
            tag='local',
        )

    if args.mode in ('hub', 'both'):
        if not args.hf_repo:
            raise ValueError("--hf_repo must be set when mode is 'hub' or 'both'")
        results['hub'] = evaluate_model(
            args.hf_repo,
            test_texts, test_labels,
            label2id, id2label,
            tag='hub',
        )

    # Comparison (mode=both)
    if args.mode == 'both' and len(results) == 2:
        local_acc = results['local']['accuracy']
        hub_acc   = results['hub']['accuracy']
        print("\n" + "="*60)
        print("Comparison: Local vs. HuggingFace Hub")
        print("="*60)
        print(f"  Local : {local_acc:.4f}")
        print(f"  Hub   : {hub_acc:.4f}")
        print(f"  Δ     : {abs(local_acc - hub_acc):.4f}")

        comparison_path = os.path.join(RESULTS_DIR, 'comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump({'local': results['local'], 'hub': results['hub']}, f, indent=2)

        plot_accuracy_comparison(results, RESULTS_DIR)

    logger.info("Evaluation complete. Plots saved to %s/", RESULTS_DIR)


if __name__ == '__main__':
    main()
