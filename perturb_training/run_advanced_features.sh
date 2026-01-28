#!/bin/bash
# Advanced: Asymmetric TR penalty + sigma clamping

python train_perturb.py \
  --base /lixiao/shenjiarun/model_from_hf/Qwen2.5-Math-7B \
  --post /lixiao/shenjiarun/model_from_hf/DeepSeek-R1-Distill-Qwen-7B \
  --data /lixiao/shenjiarun/dataset/deepscaler.json \
  --objective two_teacher \
  --gamma 0.6 \
  --entropy-beta 0.01 \
  --use-asymmetric-tr \
  --extrapolation-penalty 2.0 \
  --clamp-sigma \
  --steps 2000 \
  --track-entropy \
  --save-prefix obj_b_advanced