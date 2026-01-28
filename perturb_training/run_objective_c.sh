# run_objective_c.sh
#!/bin/bash
# Objective C: Reward warm-start + KL-to-reference (most aligned with GRPO)

python train_perturb.py \
  --base /lixiao/shenjiarun/model_from_hf/Qwen2.5-Math-7B \
  --post /lixiao/shenjiarun/model_from_hf/DeepSeek-R1-Distill-Qwen-7B \
  --data /lixiao/shenjiarun/dataset/deepscaler.json \
  --objective reward_warmstart \
  --max-len 8192 \
  --batch-size 1 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --kl-lambda 0.01 \
  --reward-type ce_based \
  --length-penalty 0.001 \
  --steps 2000 \
  --log-every 10 \
  --dtype bf16 \
  --device-map auto \
  --track-entropy \
  --save-prefix obj_c_reward

# Sweep KL penalty
for kl in 0.001 0.01 0.05; do
  python train_perturb.py \
    --base /lixiao/shenjiarun/model_from_hf/Qwen2.5-Math-7B \
    --post /lixiao/shenjiarun/model_from_hf/DeepSeek-R1-Distill-Qwen-7B \
    --data /lixiao/shenjiarun/dataset/deepscaler.json \
    --objective reward_warmstart \
    --kl-lambda $kl \
    --steps 2000 \
    --save-prefix obj_c_kl_${kl}
done