# Short Report – Assignment 3

## 1. Model Selection

**Model chosen:** `distilbert-base-cased`

**Rationale:**
- DistilBERT is a lighter, faster distillation of BERT (≈40% fewer parameters, 60% faster) while retaining 97% of BERT's language-understanding ability on GLUE benchmarks.
- The *cased* variant is preferred for book-review text because proper nouns (author names, book titles, genre-specific capitalisation) carry meaningful signal.
- The model fits comfortably in memory, making it practical to fine-tune on a standard GPU (or even CPU for experimentation).
- It is the model chosen in the provided reference notebook, so its suitability for multi-class text classification at this scale is already established.

---

## 2. Training Summary

| Hyperparameter | Value |
|----------------|-------|
| Base model | `distilbert-base-cased` |
| Epochs | 3 |
| Train batch size | 10 |
| Eval batch size | 16 |
| Learning rate | 5 × 10⁻⁵ |
| Warmup steps | 100 |
| Weight decay | 0.01 |
| Max sequence length | 512 tokens |

**Dataset:** GoodReads reviews (UCSD), sampled across 8 genres:
`poetry`, `children`, `comics_graphic`, `fantasy_paranormal`,
`history_biography`, `mystery_thriller_crime`, `romance`, `young_adult`

Each genre used 800 train / 200 test reviews (1 000 total per genre → 6 400 train, 1 600 test).

Training was performed using the Hugging Face `Trainer` API with accuracy as the primary evaluation metric and best-model saving enabled.

---

## 3. Evaluation Comparison

| Source | Accuracy |
|--------|----------|
| Local fine-tuned checkpoint | *run `src/evaluate.py --mode local`* |
| HuggingFace Hub checkpoint | *run `src/evaluate.py --mode hub`*   |

The local and Hub checkpoints are the same saved weights, so metrics should match exactly. Any tiny floating-point difference may arise from framework version differences between local and Hub environments.

---

## 4. Challenges

| Challenge | Resolution |
|-----------|-----------|
| Large data downloads (8 gzip HTTP streams) | Added pickle cache so data is only downloaded once |
| Jupyter `%matplotlib inline` magic in scripts | Removed notebook-only magic commands from `.py` files |
| Keeping Docker image lean | Used `python:3.11-slim`; separated dev (Dockerfile) and production (Dockerfile.eval) images |
| Reproducibility across environments | Set `random.seed`, pinned dependency versions in `requirements.txt` |
