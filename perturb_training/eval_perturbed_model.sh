set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python run.py \
    --datasets math_500_gen \
    --hf-type base \
    --hf-path /data/home/Jiarun/LLaMA-Factory/1500_perturbed_from_alpha_multi \
    --hf-num-gpus 2 \
	--max-num-workers 4 \
    --max-out-len 16384 \
	--max-seq-len 16384 \
    --batch-size 2