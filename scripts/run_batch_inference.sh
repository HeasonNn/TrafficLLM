#!/bin/bash
# Batch UAD Inference Script
# Usage: bash scripts/run_batch_inference.sh [OPTIONS]

set -e
source /root/anaconda3/etc/profile.d/conda.sh
conda activate trafficllm

GPU=${GPU:-7}
WINDOW_SIZE=${WINDOW_SIZE:-50}
SAMPLES_PER_DATASET=${SAMPLES_PER_DATASET:-20}
MAX_PER_FILE=${MAX_PER_FILE:-5}
LIMIT=${LIMIT:-0}

echo "============================================================"
echo "UAD Batch Inference Configuration"
echo "============================================================"
echo "GPU: $GPU"
echo "Window size: $WINDOW_SIZE"
echo "Samples per dataset: $SAMPLES_PER_DATASET"
echo "Max samples per file: $MAX_PER_FILE"
echo "Limit: $LIMIT (0 = no limit)"
echo "============================================================"

cd /root/home/bihaisong/TrafficLLM

CUDA_VISIBLE_DEVICES=$GPU python scripts/batch_uad_inference.py \
    --data_dir data/ \
    --checkpoint models/chatglm2/peft/uad-balanced-v1/checkpoint-8000 \
    --output_dir results/uad_batch_$(date +%Y%m%d_%H%M%S)/ \
    --window_size $WINDOW_SIZE \
    --samples_per_dataset $SAMPLES_PER_DATASET \
    --max_samples_per_file $MAX_PER_FILE \
    --limit $LIMIT \
    --gpu $GPU
