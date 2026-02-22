"""
data.py - Data loading and preprocessing for the GoodReads book genre classification task.
Downloads reviews from the UCSD GoodReads dataset (by genre) and prepares them for training.
"""

import gzip
import json
import random
import pickle
import requests
import os
from typing_extensions import Literal

# ---------------------------------------------------------------------------
# Dataset URLs (GoodReads reviews by genre)
# Source: https://mengtingwan.github.io/data/goodreads.html#datasets
# ---------------------------------------------------------------------------

GENRE_URL_DICT = {
    'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz',
}


def load_reviews(
    url: str,
    head: int = 10000,
    sample_size: int = 2000
) -> list[str]:
    """
    Stream reviews from a remote gzipped JSON file, returning a random sample.

    Args:
        url: URL to the .json.gz file.
        head: Maximum number of lines to read from the file.
        sample_size: Number of reviews to randomly sample from the read lines.

    Returns:
        List of review text strings.
    """
    reviews = []
    count = 0

    response = requests.get(url, stream=True, timeout=60)
    print(f"  HTTP {response.status_code} — reading data …")

    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_genres(
    genre_url_dict: dict[Literal[GENRE_URL_DICT.keys()], str] | None = None,
    head: int = 10000,
    sample_size: int = 2000,
    cache_path: str = 'genre_reviews_dict.pickle',
) -> dict[str, list[str]]:
    """
    Load (or restore from cache) reviews for every genre.

    Args:
        genre_url_dict: Mapping of genre name -> URL. Uses GENRE_URL_DICT by default.
        head: Max lines to read per genre file.
        sample_size: Reviews to sample per genre.
        cache_path: Path where the result is cached (pickle).

    Returns:
        dict {genre_name: [review_text, …]}
    """
    if genre_url_dict is None:
        genre_url_dict = GENRE_URL_DICT

    if os.path.exists(cache_path):
        print(f"Loading cached reviews from {cache_path} …")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    genre_reviews_dict = {}
    for genre, url in genre_url_dict.items():
        print(f"Loading reviews for genre: {genre}")
        genre_reviews_dict[genre] = load_reviews(url, head=head, sample_size=sample_size)

    with open(cache_path, 'wb') as f:
        pickle.dump(genre_reviews_dict, f)
    print(f"Reviews cached to {cache_path}")

    return genre_reviews_dict


def make_train_test_split(
    genre_reviews_dict: dict[str, list[str]],
    per_genre: int = 1000,
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Build train/test splits from the per-genre review dict.

    Args:
        genre_reviews_dict: {genre: [review_text, …]}
        per_genre: Number of reviews to use per genre (sampled).
        train_ratio: Fraction of reviews used for training.
        seed: Random seed for reproducibility.

    Returns:
        Tuple (train_texts, train_labels, test_texts, test_labels)
    """
    random.seed(seed)

    train_texts, train_labels = [], []
    test_texts,  test_labels  = [], []

    n_train = int(per_genre * train_ratio)

    for genre, reviews in genre_reviews_dict.items():
        reviews = random.sample(
            reviews,
            min(per_genre, len(reviews))
        )
        for r in reviews[:n_train]:
            train_texts.append(r)
            train_labels.append(genre)
        for r in reviews[n_train:]:
            test_texts.append(r)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    data = load_all_genres(sample_size=100, head=500)
    tr_t, tr_l, te_t, te_l = make_train_test_split(data, per_genre=50)
    print(f"Train: {len(tr_t)} samples, Test: {len(te_t)} samples")
    print(f"Genres: {set(tr_l)}")
