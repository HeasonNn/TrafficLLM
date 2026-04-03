#!/bin/bash
# UAD 200K 训练脚本
# 预计时间: ~21小时

set -e
source /root/anaconda3/etc/profile.d/conda.sh
conda activate trafficllm

cd /root/home/bihaisong/TrafficLLM

GPU_ID=7
TRAIN_FILE=datasets/uad_200k/tllm/uad_200k_train.json
VALID_FILE=datasets/uad_200k/tllm/uad_200k_valid.json
OUTPUT_DIR=models/chatglm2/peft/uad-200k-v1

# 训练参数
MAX_STEPS=200000
LEARNING_RATE=0.005
WARMUP_STEPS=2000
SAVE_STEPS=20000
BATCH_SIZE=1

echo "============================================================"
echo "UAD 200K Training"
echo "============================================================"
echo "GPU: $GPU_ID"
echo "Train file: $TRAIN_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Max steps: $MAX_STEPS"
echo "Learning rate: $LEARNING_RATE"
echo "Warmup: $WARMUP_STEPS"
echo "Save steps: $SAVE_STEPS"
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU_ID python dual-stage-tuning/main.py \
    --do_train \
    --train_file $TRAIN_FILE \
    --validation_file $VALID_FILE \
    --prompt_column instruction \
    --input_column input \
    --response_column output \
    --overwrite_cache \
    --cache_dir cache \
    --model_name_or_path models/chatglm2/chatglm2-6b \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 32 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps $MAX_STEPS \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    --logging_steps 100 \
    --learning_rate $LEARNING_RATE \
    --pre_seq_len 128 \
    --warmup_steps $WARMUP_STEPS

echo "============================================================"
echo "Training completed!"
echo "Checkpoint: $OUTPUT_DIR"
echo "============================================================"
