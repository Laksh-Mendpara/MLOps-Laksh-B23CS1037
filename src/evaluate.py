"""
evaluate.py  –  Evaluate a fine-tuned DistilBERT genre-classification model.

Modes
-----
  local Load the model from a local directory (default).
  hub Load the model from a Hugging Face Hub repository.

Usage examples
--------------
  # Evaluate local model
  python src/evaluate.py --mode local --model_path ./distilbert-reviews-genres

  # Evaluate model from Hugging Face Hub
  python src/evaluate.py --mode hub --hf_repo <HF_REPO>

  # Compare both
  python src/evaluate.py --mode both \
      --model_path ./distilbert-reviews-genres \
      --hf_repo <HF_REPO>
"""

import os
import json
import argparse
import logging

import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report, accuracy_score

from data import load_all_genres, make_train_test_split
from utils import MyDataset, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)

RESULTS_DIR = './results'
REVIEWS_CACHE_PATH  = './genre_reviews_dict.pickle'
MAX_LENGTH = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned DistilBERT')
    parser.add_argument('--mode', choices=['local', 'hub', 'both'], default='local',
                        help='Where to load the model from')
    parser.add_argument('--model_path', default='./distilbert-reviews-genres',
                        help='Path to local fine-tuned model directory')
    parser.add_argument('--hf_repo', default="Laksh-Mendpara/MLOps-Assignment-3",
                        help='HuggingFace Hub repo id (e.g. username/repo)')
    parser.add_argument('--sample_size', type=int, default=2000,
                        help='Reviews to sample per genre (for data loading)')
    parser.add_argument('--per_genre', type=int, default=1000,
                        help='Reviews per genre for the evaluation split')
    return parser.parse_args()


def build_test_dataset(
    sample_size: int,
    per_genre: int
) -> tuple[list[str], list[int], dict[str, int], dict[int, str]]:
    """Load data, tokenize, and return (test_dataset, test_labels, id2label)."""
    logger.info("Loading review data …")
    genre_reviews_dict = load_all_genres(
        sample_size=sample_size,
        cache_path=REVIEWS_CACHE_PATH,
    )

    _, train_labels_all, test_texts, test_labels = make_train_test_split(
        genre_reviews_dict,
        per_genre=per_genre,
    )

    unique_labels = sorted(set(train_labels_all))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label  = {idx: label for label, idx in label2id.items()}

    return test_texts, test_labels, label2id, id2label


def evaluate_model(
    model_source: str,
    test_texts: list[str],
    test_labels: list[int],
    label2id: dict[str, int],
    id2label: dict[int, str],
    tag: str = 'model'
) -> dict[str, float]:
    """
    Load *model_source* (local path or HF Hub repo id), run evaluation,
    print classification report, and save metrics JSON.

    Returns dict of metrics.
    """
    logger.info("[%s] Loading tokenizer and model from: %s", tag, model_source)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_source)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DistilBertForSequenceClassification.from_pretrained(model_source).to(device)

    # Tokenise test set
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )
    test_labels_encoded = [label2id[y] for y in test_labels]
    test_dataset = MyDataset(test_encodings, test_labels_encoded)

    # Build a minimal Trainer just for prediction
    eval_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, 'tmp_eval'),
        per_device_eval_batch_size=16,
        report_to=[],
        no_cuda=(device == 'cpu'),
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

    # Detailed prediction
    pred_output = trainer.predict(test_dataset)
    pred_ids = pred_output.predictions.argmax(-1).flatten().tolist()
    predicted_labels = [id2label[i] for i in pred_ids]

    report = classification_report(test_labels, predicted_labels)
    print(f"\n{'='*60}")
    print(f"Classification Report — {tag}")
    print('='*60)
    print(report)

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

    return metrics


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

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
            raise ValueError("--hf_repo must be specified when mode is 'hub' or 'both'")
        results['hub'] = evaluate_model(
            args.hf_repo,
            test_texts, test_labels,
            label2id, id2label,
            tag='hub',
        )

    # ------------------------------------------------------------------
    # Comparison table when both modes are evaluated
    # ------------------------------------------------------------------
    if args.mode == 'both' and len(results) == 2:
        local_acc = results['local']['accuracy']
        hub_acc = results['hub']['accuracy']
        print("\n" + "="*60)
        print("Comparison: Local vs. HuggingFace Hub model")
        print("="*60)
        print(f"  Local model accuracy : {local_acc:.4f}")
        print(f"  Hub   model accuracy : {hub_acc:.4f}")
        print(f"  Difference           : {abs(local_acc - hub_acc):.4f}")

        comparison_path = os.path.join(RESULTS_DIR, 'comparison.json')
        with open(comparison_path, 'w') as f:
            json.dump(
                {'local': results['local'], 'hub': results['hub']},
                f, indent=2
            )
        logger.info("Comparison saved to %s", comparison_path)

    logger.info("Evaluation complete.")


if __name__ == '__main__':
    main()
