"""
utils.py - Utility functions and classes for the DistilBERT genre classification task.
"""

import torch
from sklearn.metrics import accuracy_score


class MyDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset wrapper for Hugging Face tokenizer output."""

    def __init__(
        self,
        encodings: dict[str, list[int]],
        labels: list[int]
    ) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def compute_metrics(pred: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute accuracy metric for the Trainer API."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}
