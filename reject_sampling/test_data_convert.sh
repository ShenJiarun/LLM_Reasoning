python reject_sampling/format_pairwise_dataset.py \
	--input-accept accept_data.jsonl \
	--input-reject reject_data.jsonl \
	--output-format implicit_prompt_preference \
	--output-path reward_pairwise_dataset.parquet
