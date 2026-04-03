#!/usr/bin/env python3
# Fix encoding issue for numpy/sklearn
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

"""Batch UAD inference - relies on external CUDA_VISIBLE_DEVICES."""

# Fix encoding issues BEFORE other imports
import os
import locale
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('LANG', 'en_US.UTF-8')
os.environ.setdefault('LC_ALL', 'en_US.UTF-8')
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    pass


import argparse
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)


@dataclass
class Sample:
    dataset: str
    file_name: str
    window_id: int
    tokens: str
    gold_label: str
    flow_label: str


def read_data_file(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8', errors='ignore').splitlines() if line.strip()]


def read_label_file(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore').strip()


def get_flow_label(label_text: str) -> str:
    return 'abnormal' if '1' in label_text else 'normal'


def get_window_label(label_text: str, start: int, end: int) -> str:
    return 'abnormal' if '1' in label_text[start:end] else 'normal'


def sample_windows(data_lines: List[str], label_text: str, window_size: int, max_samples: int, seed: int) -> List[Tuple[int, str, str]]:
    if len(data_lines) < window_size:
        window_size = len(data_lines)
    if window_size == 0:
        return []
    
    max_start = len(data_lines) - window_size + 1
    if max_start <= 0:
        return [(0, ' '.join(data_lines), get_window_label(label_text, 0, len(data_lines)))]
    
    all_starts = list(range(0, max_start, window_size))
    if len(all_starts) > max_samples:
        random.Random(seed).shuffle(all_starts)
        all_starts = all_starts[:max_samples]
    
    return [(i, ' '.join(data_lines[s:s+window_size]), get_window_label(label_text, s, s+window_size)) 
            for i, s in enumerate(all_starts)]


def discover_data_files(data_dir: Path) -> Dict[str, List[Path]]:
    datasets = defaultdict(list)
    for data_path in sorted(data_dir.rglob('*.data')):
        if data_path.with_suffix('.label').exists():
            datasets[data_path.parent.name].append(data_path)
    return dict(datasets)


def build_samples(data_dir: Path, window_size: int, samples_per_dataset: int, max_samples_per_file: int, seed: int) -> List[Sample]:
    datasets = discover_data_files(data_dir)
    all_samples = []
    
    for dataset_name, data_files in sorted(datasets.items()):
        dataset_samples = []
        samples_per_file = min(max(1, samples_per_dataset // len(data_files)), max_samples_per_file)
        
        for data_path in data_files:
            label_path = data_path.with_suffix('.label')
            data_lines = read_data_file(data_path)
            label_text = read_label_file(label_path)
            
            if len(data_lines) != len(label_text):
                continue
            
            windows = sample_windows(data_lines, label_text, window_size, samples_per_file, hash(data_path.name) ^ seed)
            flow_label = get_flow_label(label_text)
            
            for window_id, tokens, window_label in windows:
                dataset_samples.append(Sample(dataset_name, data_path.stem, window_id, tokens, window_label, flow_label))
        
        if len(dataset_samples) > samples_per_dataset:
            random.Random(seed).shuffle(dataset_samples)
            dataset_samples = dataset_samples[:samples_per_dataset]
        
        all_samples.extend(dataset_samples)
        print(f'  {dataset_name}: {len(dataset_samples)} samples from {len(data_files)} files', flush=True)
    
    return all_samples


def load_model(checkpoint_path: str, model_path: str):
    """Load model - CUDA_VISIBLE_DEVICES must be set externally."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    
    # DO NOT set CUDA_VISIBLE_DEVICES here - must be set before Python starts
    gpu_env = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    print(f'CUDA_VISIBLE_DEVICES: {gpu_env}', flush=True)
    
    print(f'Loading model from {model_path}...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
    
    model = AutoModel.from_pretrained(
        model_path,
        config=model_config,
        trust_remote_code=True,
        load_in_8bit=True,
        device_map='auto',
        low_cpu_mem_usage=True,
    )
    
    print(f'Loading P-Tuning checkpoint from {checkpoint_path}...', flush=True)
    prefix_state_dict = torch.load(os.path.join(checkpoint_path, 'pytorch_model.bin'), map_location='cpu')
    new_prefix_state_dict = {k[len('transformer.prefix_encoder.'):]: v for k, v in prefix_state_dict.items() 
                             if k.startswith('transformer.prefix_encoder.')}
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model.transformer.prefix_encoder.float()
    model = model.eval()
    
    return tokenizer, model


def run_inference(model, tokenizer, prompt: str) -> str:
    import torch
    with torch.no_grad():
        response, _ = model.chat(tokenizer, prompt, history=[])
    pred = response.strip().lower()
    return pred if pred in ['normal', 'abnormal'] else 'normal'


def compute_metrics(predictions: List[Dict]) -> Dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    golds = [1 if p['gold_label'] == 'abnormal' else 0 for p in predictions]
    preds = [1 if p['pred_label'] == 'abnormal' else 0 for p in predictions]
    
    if len(set(golds)) < 2:
        return {'accuracy': accuracy_score(golds, preds), 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    return {
        'accuracy': float(accuracy_score(golds, preds)),
        'f1': float(f1_score(golds, preds, zero_division=0)),
        'precision': float(precision_score(golds, preds, zero_division=0)),
        'recall': float(recall_score(golds, preds, zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser(description='Batch UAD inference')
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--model_path', type=str, default='/root/home/bihaisong/TrafficLLM/models/chatglm2/chatglm2-6b')
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--samples_per_dataset', type=int, default=20)
    parser.add_argument('--max_samples_per_file', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    
    # DO NOT set os.environ['CUDA_VISIBLE_DEVICES'] here!
    
    print('\n' + '='*60, flush=True)
    print('UAD Batch Inference', flush=True)
    print('='*60, flush=True)
    print(f'Window size: {args.window_size}', flush=True)
    print(f'Samples per dataset: {args.samples_per_dataset}', flush=True)
    print('\nDiscovering data files...', flush=True)
    
    samples = build_samples(args.data_dir, args.window_size, args.samples_per_dataset, args.max_samples_per_file, args.seed)
    
    if args.limit > 0:
        samples = samples[:args.limit]
    
    print(f'\nTotal samples: {len(samples)}', flush=True)
    
    tokenizer, model = load_model(args.checkpoint, args.model_path)
    print('Model loaded!\n', flush=True)
    
    UAD_PROMPT = 'You are performing universal anomaly detection for encrypted traffic.\nDetermine whether the following traffic sequence is normal or abnormal.\nRespond with only one word: normal or abnormal.'
    
    predictions = []
    correct = 0
    
    print('Starting inference...', flush=True)
    for i, sample in enumerate(samples):
        prompt = f'{UAD_PROMPT}\n{sample.tokens}'
        pred_label = run_inference(model, tokenizer, prompt)
        
        is_correct = pred_label == sample.gold_label
        if is_correct:
            correct += 1
        
        predictions.append({
            'dataset': sample.dataset,
            'file_name': sample.file_name,
            'window_id': sample.window_id,
            'gold_label': sample.gold_label,
            'pred_label': pred_label,
            'flow_label': sample.flow_label,
            'correct': is_correct,
        })
        
        status = 'OK' if is_correct else 'WRONG'
        print(f'[{i+1}/{len(samples)}] {status} {sample.dataset}/{sample.file_name}: pred={pred_label}, gold={sample.gold_label}', flush=True)
    
    overall_metrics = compute_metrics(predictions)
    overall_metrics.update({'total_samples': len(predictions), 'correct': correct, 'accuracy_by_count': correct / len(predictions) if predictions else 0})
    
    by_dataset = defaultdict(list)
    for p in predictions:
        by_dataset[p['dataset']].append(p)
    
    per_dataset_metrics = {ds: {**compute_metrics(preds), 'total': len(preds), 'correct': sum(1 for p in preds if p['correct'])} 
                          for ds, preds in sorted(by_dataset.items())}
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'metrics_summary.json').write_text(json.dumps(overall_metrics, indent=2), encoding='utf-8')
    (args.output_dir / 'per_dataset_metrics.json').write_text(json.dumps(per_dataset_metrics, indent=2), encoding='utf-8')
    (args.output_dir / 'predictions.jsonl').write_text('\n'.join(json.dumps(p, ensure_ascii=False) for p in predictions), encoding='utf-8')
    
    print('\n' + '='*60, flush=True)
    print('RESULTS SUMMARY', flush=True)
    print('='*60, flush=True)
    print(f'Total: {len(predictions)}, Correct: {correct}, Acc: {overall_metrics["accuracy"]:.4f}', flush=True)
    print('Per-dataset:', flush=True)
    for ds, m in sorted(per_dataset_metrics.items()):
        print(f'  {ds}: acc={m["accuracy"]:.4f}, f1={m["f1"]:.4f}, n={m["total"]}', flush=True)
    print(f'\nResults: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
