# REPRODUCE

## Overview

This document describes the currently workflows:

1. Streaming UAD evaluation
2. UAD training data preparation
3. UAD fine-tuning

## Assumptions

Before running commands, make sure:

- you are inside the project root
- the required Python environment is activated
- the base model directory is available
- the UAD checkpoint directory is available
- the dataset directory follows the expected layout

In examples below, use these placeholders as needed:

- `<PROJECT_ROOT>`: repository root directory
- `<PY_ENV_SETUP>`: environment activation command(s)
- `<GPU_ENV>`: optional GPU-related environment variables
- `<BASE_MODEL_DIR>`: base model path
- `<UAD_CHECKPOINT_DIR>`: UAD checkpoint path
- `<DATA_ROOT>`: dataset root directory

A typical session looks like:

```bash
<PY_ENV_SETUP>
cd <PROJECT_ROOT>
<GPU_ENV>
```

## 1. Streaming UAD evaluation

### Required inputs

- base model directory: `<BASE_MODEL_DIR>`
- UAD checkpoint directory: `<UAD_CHECKPOINT_DIR>`
- dataset root laid out as:

```text
<DATA_ROOT>/
  <dataset_name>/
    *.data
    *.label
```

`scripts/streaming_uad_evaluation.py` expects dataset subdirectories under `--data_dir`.

### Example: build a small smoke input tree

If you already have a small dataset such as `DoHBrw_smoke`, you can prepare an input tree like this:

```bash
rm -rf results/streaming_input_smoke
mkdir -p results/streaming_input_smoke/DoHBrw_smoke
ln -sf <PROJECT_ROOT>/data/DoHBrw_smoke/*.data results/streaming_input_smoke/DoHBrw_smoke/
ln -sf <PROJECT_ROOT>/data/DoHBrw_smoke/*.label results/streaming_input_smoke/DoHBrw_smoke/
```

### Run streaming smoke evaluation

```bash
python scripts/streaming_uad_evaluation.py \
  --data_dir <PROJECT_ROOT>/results/streaming_input_smoke \
  --checkpoint <UAD_CHECKPOINT_DIR> \
  --output_dir <PROJECT_ROOT>/results/streaming_smoke_test \
  --window_size 10 \
  --samples_per_file 2 \
  --device cuda
```

### Outputs

The streaming evaluation writes:

- `summary.json`
- `predictions.jsonl`
- `results_by_dataset.csv`
- `results_by_category.csv`

### View results

```bash
ls -lh results/streaming_smoke_test
python -c "import json; from pathlib import Path; print(json.loads(Path(results/streaming_smoke_test/summary.json).read_text()))"
```

### Run on a full dataset tree

```bash
python scripts/streaming_uad_evaluation.py \
  --data_dir <DATA_ROOT> \
  --checkpoint <UAD_CHECKPOINT_DIR> \
  --output_dir <PROJECT_ROOT>/results/streaming_full_eval \
  --window_size 50 \
  --samples_per_file 100 \
  --device cuda
```

## 2. Prepare 200K UAD training data

Training-data entrypoint:

- `scripts/prepare_uad_200k.py`

Run:

```bash
python scripts/prepare_uad_200k.py
```

Expected outputs:

- `datasets/uad_200k/tllm/uad_200k_train.json`
- `datasets/uad_200k/tllm/uad_200k_valid.json`
- `datasets/uad_200k/tllm/uad_200k_test.json`

## 3. Fine-tune the UAD model

Fine-tuning entrypoint:

- `scripts/train_uad_200k.sh`

Run:

```bash
bash scripts/train_uad_200k.sh
```

This script expects:

- `datasets/uad_200k/tllm/uad_200k_train.json`
- `datasets/uad_200k/tllm/uad_200k_valid.json`
- the base model directory under the project
- an output directory under `models/chatglm2/peft/`

Notes:

- the current shell script may still contain project-specific defaults
- training data must already exist before running the script
- if you move the repo to a different environment, check the script arguments and paths first

## 4. Retained regression tests

Run the retained test suite with:

```bash
python -m unittest \
  tests.test_attack_groups \
  tests.test_uad_inference \
  tests.test_convert_data_label_to_json \
  tests.test_uad_converter \
  tests.test_inference_quantization -v
```

These tests cover the retained functionality after cleanup.
