#!/usr/bin/env bash
# Serves a checkpoint with vLLM (0.26.0) behind an OpenAI-compatible endpoint, with the settings
# every reported score was measured under: one image a prompt, 8192 tokens, FlashInfer for the
# language model and SDPA for the vision encoder. The 4B was served at 0.3 of an H200's memory.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/serve.sh [CHECKPOINT] [PORT] [GPU_MEMORY_UTILIZATION]
set -euo pipefail
CK=${1:-checkpoints/script2play-4b}
PORT=${2:-8000}
UTIL=${3:-0.3}
exec vllm serve "$CK" --served-model-name script2play --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code --max-model-len 8192 --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":1,"video":0}' --gpu-memory-utilization "$UTIL" --max-num-seqs 32 \
  --attention-backend FLASHINFER --mm-encoder-attn-backend TORCH_SDPA
