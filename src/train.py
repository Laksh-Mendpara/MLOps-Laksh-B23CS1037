"""
train.py - Fine-tune DistilBERT on the GoodReads book-genre classification task.

Usage:
    python src/train.py [--epochs N] [--batch_size B] [--output_dir PATH] [--hf_repo REPO_ID]

The script:
  1. Downloads / loads the GoodReads review data.
  2. Tokenises the text with DistilBertTokenizerFast.
  3. Fine-tunes DistilBertForSequenceClassification with the Hugging Face Trainer API.
  4. Saves the model locally and (optionally) pushes it to the HF Hub.
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

from data import load_all_genres, make_train_test_split
from utils import MyDataset, compute_metrics
from huggingface_hub import HfApi, login
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
MODEL_NAME = 'distilbert-base-cased'
CACHED_MODEL_DIR = './distilbert-reviews-genres'
RESULTS_DIR = './results'
LOGS_DIR = './logs'
CACHE_FILE = './genre_reviews_dict.pickle'
MAX_LENGTH = 512
REVIEWS_CACHE_PATH = CACHE_FILE


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune DistilBERT for genre classification')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Train batch size per device')
    parser.add_argument('--eval_batch', type=int, default=16, help='Eval batch size per device')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default=CACHED_MODEL_DIR, help='Where to save the model')
    parser.add_argument('--hf_repo', type=str, default='Laksh-Mendpara/MLOps-Assignment-3', help='HuggingFace Hub repo id (e.g. username/repo)')
    parser.add_argument('--sample_size', type=int, default=2000, help='Reviews to sample per genre')
    parser.add_argument('--per_genre', type=int, default=1000, help='Reviews per genre for train/test split')
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------
    logger.info("Setting up environment …")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("Using device: %s", device)
    HF_TOKEN = os.environ.get('HF_TOKEN')
    if HF_TOKEN is None:
        raise ValueError("Please set the HF_TOKEN environment variable")
    else:
        HF_TOKEN = HF_TOKEN.strip()

    # Check validity
    try:
        login(HF_TOKEN)
        _hf_username = HfApi().whoami()
    except Exception as e:
        raise ValueError("Invalid HF_TOKEN") from e
    logger.info("Using HF user: %s", _hf_username)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading review data …")
    genre_reviews_dict = load_all_genres(
        sample_size=args.sample_size,
        cache_path=REVIEWS_CACHE_PATH,
    )

    train_texts, train_labels, test_texts, test_labels = make_train_test_split(
        genre_reviews_dict,
        per_genre=args.per_genre,
    )
    logger.info("Train: %d  Test: %d", len(train_texts), len(test_texts))

    # ------------------------------------------------------------------
    # 2. Tokenise
    # ------------------------------------------------------------------
    logger.info("Tokenising …")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=MAX_LENGTH)

    # ------------------------------------------------------------------
    # 3. Label mappings
    # ------------------------------------------------------------------
    unique_labels = sorted(set(train_labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded = [label2id[y] for y in test_labels]

    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    test_dataset = MyDataset(test_encodings, test_labels_encoded)

    # Save label mappings alongside the model
    label_map_path = os.path.join(args.output_dir, 'label_map.json')
    with open(label_map_path, 'w') as f:
        json.dump({'label2id': label2id, 'id2label': {str(k): v for k, v in id2label.items()}}, f, indent=2)
    logger.info("Label map saved to %s", label_map_path)

    # ------------------------------------------------------------------
    # 4. Load pre-trained model
    # ------------------------------------------------------------------
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # ------------------------------------------------------------------
    # 5. Training arguments
    # ------------------------------------------------------------------
    os.environ['WANDB_DISABLED'] = 'true'

    training_args = TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch,
        learning_rate=args.lr,
        warmup_steps=100,
        weight_decay=0.01,
        output_dir=RESULTS_DIR,
        logging_dir=LOGS_DIR,
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=100,
        save_strategy='steps',
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        report_to=[],
    )

    # ------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training …")
    trainer.train()

    # ------------------------------------------------------------------
    # 7. Save model & tokenizer
    # ------------------------------------------------------------------
    logger.info("Saving model to %s …", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ------------------------------------------------------------------
    # 8. Evaluate and save metrics
    # ------------------------------------------------------------------
    logger.info("Running final evaluation …")
    eval_results = trainer.evaluate()
    logger.info("Eval results: %s", eval_results)

    metrics_path = os.path.join(RESULTS_DIR, 'train_eval_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    # ------------------------------------------------------------------
    # 9. (Optional) Push to Hugging Face Hub
    # ------------------------------------------------------------------
    if args.hf_repo:
        logger.info("Pushing model to HuggingFace Hub: %s …", args.hf_repo)
        trainer.push_to_hub(args.hf_repo)
        tokenizer.push_to_hub(args.hf_repo)
        logger.info("Model pushed successfully.")

    logger.info("Done.")


if __name__ == '__main__':
    main()
