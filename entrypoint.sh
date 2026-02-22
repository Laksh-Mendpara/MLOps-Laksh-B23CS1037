#!/bin/sh
# entrypoint.sh  –  Runs evaluation automatically on container startup.
# The model is pulled from the Hugging Face Hub at runtime (no model baked in).

set -e

HF_REPO="${HF_REPO:-Laksh-Mendpara/MLOps-Assignment-3}"

echo "======================================================"
echo "  Evaluating model from HuggingFace Hub: ${HF_REPO}"
echo "======================================================"

cd /app

python src/evaluate.py \
    --mode      hub \
    --hf_repo   "${HF_REPO}" \
    --sample_size 2000 \
    --per_genre   1000

echo "======================================================"
echo "  Evaluation complete. Results are in /app/results/"
echo "======================================================"
