#!/usr/bin/env python3
"""准备200K样本的UAD训练数据 - 修复版

Label文件格式：单行字符串，每个字符(0/1)对应一个数据行
"""

import json
import random
from pathlib import Path

BASE_DIR = Path('/root/home/bihaisong/TrafficLLM')
WINDOW_SIZE = 64

TARGET_SAMPLES = {
    'CICIDS2017':       {'normal': 15000, 'abnormal': 15000},
    'CICIIOT2025':      {'normal': 25000, 'abnormal': 25000},
    'CIC_APT_IIoT2024': {'normal': 10000, 'abnormal': 10000},
    'DoHBrw':           {'normal': 15000, 'abnormal': 15000},
    'hypervision':      {'normal': 25000, 'abnormal': 25000},
    'UNSW_NB15':        {'normal': 10000, 'abnormal': 10000},
}

INSTRUCTION = "You are performing universal anomaly detection for encrypted traffic.\nDetermine whether the following traffic sequence is normal or abnormal.\nRespond with only one word: normal or abnormal."


def process_file(data_path, max_normal, max_abnormal):
    """处理单个文件，返回normal和abnormal窗口列表"""
    label_path = data_path.with_suffix('.label')
    if not label_path.exists():
        return [], []
    
    normal_windows = []
    abnormal_windows = []
    
    # 读取label（单行字符串）
    labels = label_path.read_text(errors='ignore').strip()
    
    # 流式读取data行
    window_lines = []
    line_idx = 0
    
    with open(data_path, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            window_lines.append(line)
            
            if len(window_lines) >= WINDOW_SIZE:
                # 检查窗口标签
                start_idx = line_idx - WINDOW_SIZE + 1
                window_labels = labels[start_idx:line_idx+1] if start_idx >= 0 else labels[:line_idx+1]
                label = 'abnormal' if '1' in window_labels else 'normal'
                tokens = ' '.join(window_lines)
                
                if label == 'normal' and len(normal_windows) < max_normal:
                    normal_windows.append(tokens)
                elif label == 'abnormal' and len(abnormal_windows) < max_abnormal:
                    abnormal_windows.append(tokens)
                
                # 检查是否已满
                if len(normal_windows) >= max_normal and len(abnormal_windows) >= max_abnormal:
                    break
                
                window_lines = []
            
            line_idx += 1
    
    return normal_windows, abnormal_windows


def process_dataset(ds_name, target):
    """处理数据集"""
    ds_path = BASE_DIR / 'data' / ds_name
    if not ds_path.exists():
        print(f'  {ds_name}: not found')
        return [], []
    
    data_files = sorted(ds_path.glob('*.data'))
    if not data_files:
        print(f'  {ds_name}: no data files')
        return [], []
    
    all_normal = []
    all_abnormal = []
    
    per_file_normal = max(1, target['normal'] // len(data_files) + 1)
    per_file_abnormal = max(1, target['abnormal'] // len(data_files) + 1)
    
    for data_file in data_files:
        if len(all_normal) >= target['normal'] and len(all_abnormal) >= target['abnormal']:
            break
        
        remaining_normal = target['normal'] - len(all_normal)
        remaining_abnormal = target['abnormal'] - len(all_abnormal)
        
        normal, abnormal = process_file(
            data_file,
            min(per_file_normal, remaining_normal),
            min(per_file_abnormal, remaining_abnormal)
        )
        all_normal.extend(normal)
        all_abnormal.extend(abnormal)
    
    # 采样
    random.seed(42)
    if len(all_normal) > target['normal']:
        all_normal = random.sample(all_normal, target['normal'])
    if len(all_abnormal) > target['abnormal']:
        all_abnormal = random.sample(all_abnormal, target['abnormal'])
    
    print(f'  {ds_name}: {len(all_normal)} normal, {len(all_abnormal)} abnormal')
    return all_normal, all_abnormal


def main():
    output_dir = BASE_DIR / 'datasets/uad_200k/tllm'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print('='*60)
    print('Preparing 200K UAD Training Data')
    print('='*60)
    
    all_normal = []
    all_abnormal = []
    
    for ds_name, target in TARGET_SAMPLES.items():
        normal, abnormal = process_dataset(ds_name, target)
        all_normal.extend(normal)
        all_abnormal.extend(abnormal)
    
    # 合并
    all_samples = [(t, 'normal') for t in all_normal] + [(t, 'abnormal') for t in all_abnormal]
    random.shuffle(all_samples)
    
    # 划分
    n = len(all_samples)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    
    train = all_samples[:n_train]
    valid = all_samples[n_train:n_train+n_valid]
    test = all_samples[n_train+n_valid:]
    
    # 写入
    def write_jsonl(samples, path):
        with open(path, 'w') as f:
            for tokens, label in samples:
                f.write(json.dumps({'instruction': INSTRUCTION, 'input': tokens, 'output': label}, ensure_ascii=False) + '\n')
    
    write_jsonl(train, output_dir / 'uad_200k_train.json')
    write_jsonl(valid, output_dir / 'uad_200k_valid.json')
    write_jsonl(test, output_dir / 'uad_200k_test.json')
    
    print('='*60)
    print(f'Total: {n} (normal={len(all_normal)}, abnormal={len(all_abnormal)})')
    print(f'Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}')
    print(f'Output: {output_dir}')
    print('='*60)


if __name__ == '__main__':
    main()
