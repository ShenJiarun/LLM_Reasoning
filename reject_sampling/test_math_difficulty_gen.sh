POLICY_MODEL_PATH=/root/Qwen/Qwen3-1.7B
DATA_PATH="parquet@/root/math_level3to5_data_processed_without_qwen_prompt"

ROLLOUT_BATCH_SIZE=1000
N=1
iter=0

python /root/reject_sampling/rejection_sampling.py \
   --pretrain $POLICY_MODEL_PATH \
   --task reject_sampling_math_difficult \
   --max-new-tokens 2048 \
   --prompt-max-len 2048 \
   --dataset $DATA_PATH \
   --map-keys '{"prompt":"","gt_answer":"","response":""}' \
   --temperature 0.7 \
   --repetition-penalty 1.05 \
   --top-p 0.8 \
   --best-of-n $N \
   --enable-prefix-caching \
   --tp-size 1 \
   --sampling-language en \
   --iter $iter \
   --rollout-batch-size $ROLLOUT_BATCH_SIZE \
   --output-path generate_output_math_difficult_Qwen3_1.7B.jsonl
