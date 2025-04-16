data="/root/rejection_sampling_output_4096.jsonl"

python reject_sampling/rejection_sampling.py \
   --dataset $data \
   --task filter_sampling_result \
   --reward-threshold 5 \
   --output-path rejection_sampling_output_filtered.jsonl
