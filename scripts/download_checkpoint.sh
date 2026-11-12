#!/usr/bin/env bash
# Downloads a checkpoint (weights, tokenizer, chat template; 8.5 GB for the 4B) from Hugging Face.
#   bash scripts/download_checkpoint.sh HF_REPO_ID [DEST]
set -euo pipefail
REPO=${1:?usage: download_checkpoint.sh HF_REPO_ID [DEST]}
DEST=${2:-checkpoints/script2play-4b}
mkdir -p "$DEST"
huggingface-cli download "$REPO" --local-dir "$DEST"
du -sh "$DEST"
