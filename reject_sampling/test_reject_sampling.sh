POLICY_MODEL_PATH=/root/Qwen/Qwen2.5-1.5B-Instruct
data="generate_output.jsonl"

python reject_sampling/rejection_sampling.py \
   --pretrain $POLICY_MODEL_PATH \
   --task rejection_sampling \
   --map-keys '{"prompt":"prompt","gt_answer":"gt_answer","response":"output"}' \
   --use-ground-truth-answer \
   --max-new-tokens 4096 \
   --prompt-max-len 4096 \
   --dataset $data \
   --temperature 0.3 \
   --top-p 0.3 \
   --repetition-penalty 1.05 \
   --enable-prefix-caching \
   --tp-size 1 \
   --sampling-language en \
   --output-path rejection_sampling_output_4096.jsonl
