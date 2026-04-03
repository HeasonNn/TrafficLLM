#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streaming UAD evaluation - memory efficient version.

Key changes:
- Does NOT pre-collect all samples
- Processes one window at a time
- Uses generator pattern to avoid memory explosion
"""

import os
import sys

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('LANG', 'en_US.UTF-8')
os.environ.setdefault('LC_ALL', 'en_US.UTF-8')

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
from statistics import mean

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from attack_groups import match_attack_category


@dataclass
class EvalConfig:
    data_dir: Path
    checkpoint: Path
    output_dir: Path
    window_size: int = 50
    samples_per_file: int = 100
    seed: int = 42
    device: str = "cuda"


@dataclass
class Metrics:
    total: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    
    def add(self, gold: str, pred: str):
        self.total += 1
        if gold == "abnormal":
            if pred == "abnormal":
                self.tp += 1
            else:
                self.fn += 1
        else:
            if pred == "abnormal":
                self.fp += 1
            else:
                self.tn += 1
    
    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0
    
    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else 0.0
    
    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else 0.0
    
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def get_window_label(label_text: str, start: int, end: int) -> str:
    return "abnormal" if '1' in label_text[start:end] else "normal"


def extract_file_token(filename: str) -> str:
    token = re.sub(r'\.(data|label)$', '', filename, flags=re.IGNORECASE)
    token = re.sub(r'^attack_', '', token)
    token = re.sub(r'^normal_', '', token)
    return token


def count_data_lines(path: Path) -> int:
    count = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            count += 1
    return count


def stream_windows(
    config: EvalConfig
) -> Iterator[Tuple[str, str, str, str]]:
    """Stream windows one at a time - memory efficient.
    
    Yields: (dataset, file_name, tokens, gold_label)
    """
    random.seed(config.seed)
    
    for dataset_dir in sorted(config.data_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        data_files = sorted(dataset_dir.glob("*.data"))
        
        for data_file in data_files:
            label_file = data_file.with_suffix('.label')
            if not label_file.exists():
                continue
            
            file_token = extract_file_token(data_file.stem)
            
            n_lines = count_data_lines(data_file)
            if n_lines < config.window_size:
                continue
            
            max_windows = n_lines - config.window_size + 1
            
            window_indices = random.sample(
                range(max_windows),
                min(config.samples_per_file, max_windows)
            )
            
            with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
            
            label_text = label_file.read_text(encoding='utf-8', errors='ignore').strip()
            
            for window_id in window_indices:
                start = window_id
                end = window_id + config.window_size
                
                window_data = [all_lines[i].strip() for i in range(start, end)]
                tokens = " ".join(window_data)
                gold_label = get_window_label(label_text, start, end)
                
                yield dataset_name, data_file.stem, tokens, gold_label, file_token
            
            del all_lines


def load_model(checkpoint_path: str, device: str = "cuda"):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    
    ckpt_path = Path(checkpoint_path).resolve()
    base_model_path = str(ckpt_path.parent.parent.parent / "chatglm2-6b")
    
    print(f"Loading base model from {base_model_path}...")
    print(f"Checkpoint path: {ckpt_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True, pre_seq_len=128)
    
    model = AutoModel.from_pretrained(
        base_model_path,
        config=model_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    if device == "cuda":
        model = model.cuda()
    
    print(f"Loading P-Tuning checkpoint from {checkpoint_path}...")
    prefix_state_dict = torch.load(
        os.path.join(str(ckpt_path), 'pytorch_model.bin'),
        map_location='cpu'
    )
    new_prefix_state_dict = {
        k[len('transformer.prefix_encoder.'):]: v 
        for k, v in prefix_state_dict.items() 
        if k.startswith('transformer.prefix_encoder.')
    }
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model.transformer.prefix_encoder.float()
    model = model.eval()
    
    return tokenizer, model


def predict(tokenizer, model, tokens: str, device: str = "cuda") -> Tuple[str, float]:
    """Predict with probability score for AUC calculation.
    
    Returns:
        (label, prob_abnormal): label is 'normal' or 'abnormal', 
        prob_abnormal is the probability of abnormal class (0-1)
    """
    prompt = f"Analyze the following network traffic and classify it as 'normal' or 'abnormal':\n\n{tokens}\n\nClassification:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    
    first_token_scores = outputs.scores[0][0]
    probs = torch.softmax(first_token_scores, dim=-1)
    
    abnormal_tokens = tokenizer.encode("abnormal", add_special_tokens=False)
    normal_tokens = tokenizer.encode("normal", add_special_tokens=False)
    
    prob_abnormal = sum(probs[t].item() for t in abnormal_tokens if t < len(probs))
    prob_normal = sum(probs[t].item() for t in normal_tokens if t < len(probs))
    
    total = prob_abnormal + prob_normal
    if total > 0:
        prob_abnormal = prob_abnormal / total
    else:
        prob_abnormal = 0.5
    
    generated = tokenizer.decode(outputs.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    
    if "abnormal" in generated:
        return "abnormal", prob_abnormal
    else:
        return "normal", prob_abnormal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--samples_per_file", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        data_dir=Path(args.data_dir),
        checkpoint=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        window_size=args.window_size,
        samples_per_file=args.samples_per_file,
        seed=args.seed,
        device=args.device,
    )
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("UAD Streaming Evaluation (Memory Efficient)")
    print("=" * 60)
    print(f"Data directory: {config.data_dir}")
    print(f"Checkpoint: {config.checkpoint}")
    print(f"Output directory: {config.output_dir}")
    print(f"Window size: {config.window_size}")
    print(f"Samples per file: {config.samples_per_file}")
    
    total_files = sum(1 for _ in config.data_dir.rglob("*.data"))
    print(f"Total data files: {total_files}")
    
    print("\nLoading model...")
    tokenizer, model = load_model(str(config.checkpoint), config.device)
    print("Model loaded!")
    
    overall_metrics = Metrics()
    by_dataset: Dict[str, Metrics] = defaultdict(Metrics)
    by_category: Dict[str, Metrics] = defaultdict(Metrics)
    by_dataset_category: Dict[str, Metrics] = defaultdict(Metrics)
    
    y_true = []
    y_prob = []
    predictions_list = []
    
    print("\nEvaluating...")
    for dataset, file_name, tokens, gold, file_token in tqdm(stream_windows(config), desc="Samples"):
        pred_label, pred_prob = predict(tokenizer, model, tokens, config.device)
        category = match_attack_category(dataset, file_token) or "Unknown"
        
        overall_metrics.add(gold, pred_label)
        by_dataset[dataset].add(gold, pred_label)
        by_category[category].add(gold, pred_label)
        by_dataset_category[f"{dataset}|{category}"].add(gold, pred_label)
        
        y_true.append(1 if gold == "abnormal" else 0)
        y_prob.append(pred_prob)
        
        predictions_list.append({
            "dataset": dataset,
            "file": file_name,
            "category": category,
            "gold": gold,
            "pred": pred_label,
            "prob": round(pred_prob, 4),
            "correct": gold == pred_label,
        })
    
    print("\n" + "=" * 60)
    print("=== Final ===")
    print("=" * 60)
    
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        has_auc = True
    except Exception as e:
        print(f"Warning: Could not calculate AUC: {e}")
        auc_roc = 0.0
        auc_pr = 0.0
        has_auc = False
    
    print(f"Total samples: {overall_metrics.total}")
    if has_auc:
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
    print(f"Accuracy: {overall_metrics.accuracy:.4f}")
    print(f"Precision: {overall_metrics.precision:.4f}")
    print(f"Recall: {overall_metrics.recall:.4f}")
    print(f"F1: {overall_metrics.f1:.4f}")
    print(f"Confusion: TP={overall_metrics.tp}, TN={overall_metrics.tn}, FP={overall_metrics.fp}, FN={overall_metrics.fn}")
    
    print("\n--- By Dataset ---")
    for ds, m in sorted(by_dataset.items()):
        print(f"{ds}: Acc={m.accuracy:.4f}, P={m.precision:.4f}, R={m.recall:.4f}, F1={m.f1:.4f} (n={m.total})")
    
    print("\n--- By Attack Category ---")
    for cat, m in sorted(by_category.items()):
        print(f"{cat}: Acc={m.accuracy:.4f}, P={m.precision:.4f}, R={m.recall:.4f}, F1={m.f1:.4f} (n={m.total})")
    
    print("\n--- By Dataset + Category ---")
    for key, m in sorted(by_dataset_category.items()):
        print(f"{key}: Acc={m.accuracy:.4f}, P={m.precision:.4f}, R={m.recall:.4f}, F1={m.f1:.4f} (n={m.total})")
    
    print("=" * 60)
    
    overall_dict = overall_metrics.to_dict()
    overall_dict["auc_roc"] = round(auc_roc, 4)
    overall_dict["auc_pr"] = round(auc_pr, 4)
    
    summary = {
        "overall": overall_dict,
        "by_dataset": {k: v.to_dict() for k, v in by_dataset.items()},
        "by_category": {k: v.to_dict() for k, v in by_category.items()},
        "by_dataset_category": {k: v.to_dict() for k, v in by_dataset_category.items()},
    }
    
    with open(config.output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(config.output_dir / "predictions.jsonl", 'w') as f:
        for p in predictions_list:
            f.write(json.dumps(p) + "\n")
    
    import csv
    with open(config.output_dir / "results_by_dataset.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "total", "accuracy", "precision", "recall", "f1"])
        for k, v in sorted(by_dataset.items()):
            writer.writerow([k, v.total, v.accuracy, v.precision, v.recall, v.f1])
    
    with open(config.output_dir / "results_by_category.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["category", "total", "accuracy", "precision", "recall", "f1"])
        for k, v in sorted(by_category.items()):
            writer.writerow([k, v.total, v.accuracy, v.precision, v.recall, v.f1])
    
    print(f"\nResults saved to {config.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()