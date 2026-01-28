#!/bin/bash
# Objective A: CE-only baseline (mode-seeking, low entropy)
# export HF_ENDPOINT=https://hf-mirror.com

python train_perturb.py \
  --base Qwen/Qwen2.5-Math-7B \
  --post deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --data LLM_Reasoning/deepscaler.json \
  --objective ce_only \
  --max-len 8192 \
  --batch-size 1 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --ce-kl-beta 0.25 \
  --ce-tr-lambda 1e-3 \
  --steps 2000 \
  --log-every 10 \
  --dtype bf16 \
  --device-map auto \
  --track-entropy \
  --save-prefix obj_a_baseline