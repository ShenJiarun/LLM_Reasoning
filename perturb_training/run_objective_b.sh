#!/bin/bash
# Objective B: Two-teacher distillation + entropy bonus (recommended)

python train_perturb.py \
  --base /lixiao/shenjiarun/model_from_hf/Qwen2.5-Math-7B \
  --post /lixiao/shenjiarun/model_from_hf/DeepSeek-R1-Distill-Qwen-7B \
  --data /lixiao/shenjiarun/dataset/deepscaler.json \
  --objective two_teacher \
  --max-len 8192 \
  --batch-size 1 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --gamma 0.6 \
  --temp-post 2.0 \
  --temp-base 2.0 \
  --temp-student 1.0 \
  --entropy-beta 0.01 \
  --nll-lambda 0.05 \
  --steps 2000 \
  --log-every 10 \
  --dtype bf16 \
  --device-map auto \
  --track-entropy \
  --save-prefix obj_b_two_teacher

# Sweep gamma (how post-like)
for gamma in 0.4 0.6 0.8; do
  python train_perturb.py \
    --base /lixiao/shenjiarun/model_from_hf/Qwen2.5-Math-7B \
    --post /lixiao/shenjiarun/model_from_hf/DeepSeek-R1-Distill-Qwen-7B \
    --data /lixiao/shenjiarun/dataset/deepscaler.json \
    --objective two_teacher \
    --gamma $gamma \
    --entropy-beta 0.01 \
    --steps 2000 \
    --save-prefix obj_b_gamma_${gamma}
done

# Sweep entropy bonus
for beta in 0.0 0.01 0.03; do
  python train_perturb.py \
    --base /lixiao/shenjiarun/model_from_hf/Qwen2.5-Math-7B \
    --post /lixiao/shenjiarun/model_from_hf/DeepSeek-R1-Distill-Qwen-7B \
    --data /lixiao/shenjiarun/dataset/deepscaler.json \
    --objective two_teacher \
    --gamma 0.6 \
    --entropy-beta $beta \
    --steps 2000 \
    --save-prefix obj_b_entropy_${beta}
done