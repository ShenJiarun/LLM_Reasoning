python train/train_sft.py \
    --bf16 true \
    --fp16 false \
    --output_dir output_test \
    --num_train_epochs 2 \
    --model_max_length 327680 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --save_strategy "no"

#     TODO: Add deepspeed script