import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_PROMPT = "Please classify whether this traffic is benign or attack."
DEFAULT_MARKER = "packet"
LABEL_MAP = {"benign": 0, "attack": 1}
UAD_LABEL_MAP = {"normal": 0, "abnormal": 1}
UAD_TEMPLATE_V1 = (
    "You are performing universal anomaly detection for encrypted traffic.\n"
    "Determine whether the following traffic sequence is normal or abnormal.\n"
    "Respond with only one word: normal or abnormal."
)
UAD_TEMPLATES = {"uad_template_v1": UAD_TEMPLATE_V1}


@dataclass(frozen=True)
class SampleRecord:
    sample_name: str
    instruction: str
    output: str


@dataclass(frozen=True)
class UadSplitRecord:
    sample_name: str
    record: Dict[str, object]


def read_nonempty_data_lines(data_path: Path) -> List[str]:
    return [
        line.strip()
        for line in data_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]


def read_label_text(label_path: Path) -> str:
    return label_path.read_text(encoding="utf-8", errors="ignore").strip()


def build_instruction(traffic_text: str, marker_token: str, prompt: str) -> str:
    return f"Given the following traffic data {marker_token}: {traffic_text} {prompt}".strip()


def aggregate_binary_label(label_text: str) -> str:
    if not label_text:
        raise ValueError("Label text is empty.")
    invalid_chars = {char for char in label_text if char not in {"0", "1"}}
    if invalid_chars:
        raise ValueError(f"Unsupported label characters: {sorted(invalid_chars)}")
    return "attack" if "1" in label_text else "benign"


def aggregate_uad_label(label_text: str) -> str:
    return "abnormal" if aggregate_binary_label(label_text) == "attack" else "normal"


def chunk_sequence(items: Sequence[str], window_size: int) -> List[Tuple[int, Sequence[str]]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    return [(start, items[start:start + window_size]) for start in range(0, len(items), window_size)]


def maybe_subsample_records(records, max_samples: int, sample_seed: int):
    if max_samples <= 0 or len(records) <= max_samples:
        return records
    subset = list(records)
    random.Random(sample_seed).shuffle(subset)
    return subset[:max_samples]


def serialize_window(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def build_sample_records(input_dir: Path, marker_token: str, prompt: str, window_size: int = 0) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    effective_window = window_size if window_size and window_size > 0 else None

    for data_path in sorted(input_dir.rglob("*.data")):
        label_path = data_path.with_suffix(".label")
        if not label_path.exists():
            continue

        data_lines = read_nonempty_data_lines(data_path)
        label_text = read_label_text(label_path)
        if not data_lines:
            continue
        if len(data_lines) != len(label_text):
            continue

        if effective_window is None:
            windows = [(0, data_lines)]
            label_windows = [(0, label_text)]
        else:
            windows = chunk_sequence(data_lines, effective_window)
            label_windows = chunk_sequence(label_text, effective_window)

        for index, ((_, line_window), (_, label_window)) in enumerate(zip(windows, label_windows)):
            traffic_text = serialize_window(line_window)
            output = aggregate_binary_label("".join(label_window))
            instruction = build_instruction(traffic_text, marker_token, prompt)
            sample_name = data_path.stem if effective_window is None else f"{data_path.stem}__w{index:04d}"
            records.append(
                SampleRecord(
                    sample_name=sample_name,
                    instruction=instruction,
                    output=output,
                )
            )
    return records


def build_uad_record(
    dataset: str,
    split: str,
    flow_id: str,
    window_id: int,
    tokens: Sequence[str],
    window_label_text: str,
    flow_label_text: str,
    template_name: str,
    window_start: int,
    window_end: int,
    sequence_length: int,
) -> Dict[str, object]:
    if template_name not in UAD_TEMPLATES:
        raise ValueError(f"Unsupported UAD template: {template_name}")
    abnormal_count = window_label_text.count("1")
    window_size = max(window_end - window_start, 0)
    abnormal_ratio = float(abnormal_count / window_size) if window_size else 0.0
    return {
        "task": "UAD",
        "dataset": dataset,
        "split": split,
        "flow_id": flow_id,
        "window_id": window_id,
        "instruction": UAD_TEMPLATES[template_name],
        "input": serialize_window(tokens),
        "label": aggregate_uad_label(window_label_text),
        "flow_label": aggregate_uad_label(flow_label_text),
        "meta": {
            "window_start": window_start,
            "window_end": window_end,
            "window_size": window_size,
            "original_sequence_length": sequence_length,
            "abnormal_count_in_window": abnormal_count,
            "abnormal_ratio_in_window": abnormal_ratio,
        },
    }


def export_uad_sample(record: Dict[str, object]) -> Dict[str, str]:
    return {
        "instruction": str(record["instruction"]),
        "input": str(record["input"]),
        "output": str(record["label"]),
    }


def build_uad_records(input_dir: Path, dataset_name: str, template_name: str, window_size: int = 0) -> List[UadSplitRecord]:
    records: List[UadSplitRecord] = []
    effective_window = window_size if window_size and window_size > 0 else None

    for data_path in sorted(input_dir.rglob("*.data")):
        label_path = data_path.with_suffix(".label")
        if not label_path.exists():
            continue

        data_lines = read_nonempty_data_lines(data_path)
        label_text = read_label_text(label_path)
        if not data_lines or len(data_lines) != len(label_text):
            continue

        if effective_window is None:
            windows = [(0, data_lines)]
            label_windows = [(0, label_text)]
        else:
            windows = chunk_sequence(data_lines, effective_window)
            label_windows = chunk_sequence(label_text, effective_window)

        for index, ((window_start, line_window), (_, label_window)) in enumerate(zip(windows, label_windows)):
            sample_name = data_path.stem if effective_window is None else f"{data_path.stem}__w{index:04d}"
            record = build_uad_record(
                dataset=dataset_name,
                split="pending",
                flow_id=data_path.stem,
                window_id=index,
                tokens=list(line_window),
                window_label_text="".join(label_window),
                flow_label_text=label_text,
                template_name=template_name,
                window_start=window_start,
                window_end=window_start + len(line_window),
                sequence_length=len(data_lines),
            )
            records.append(UadSplitRecord(sample_name=sample_name, record=record))
    return records


def split_records(records, train_ratio: float, valid_ratio: float, seed: int):
    if not records:
        raise ValueError("No valid .data/.label pairs found.")
    if train_ratio <= 0 or valid_ratio < 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Ratios must satisfy train_ratio > 0, valid_ratio >= 0, and train_ratio + valid_ratio < 1.")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    if train_end <= 0:
        train_end = 1
    if valid_ratio > 0 and valid_end <= train_end and total - train_end > 1:
        valid_end = train_end + 1
    if valid_end >= total:
        valid_end = total - 1

    train_records = shuffled[:train_end]
    valid_records = shuffled[train_end:valid_end]
    test_records = shuffled[valid_end:]

    if not test_records:
        if valid_records:
            test_records = [valid_records.pop()]
        elif len(train_records) > 1:
            test_records = [train_records.pop()]
        else:
            raise ValueError("Need at least two records to create a test split.")

    return train_records, valid_records, test_records


def write_jsonl(path: Path, records: List[SampleRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({"instruction": record.instruction, "output": record.output}, ensure_ascii=False))
            handle.write("\n")


def write_jsonl_dicts(path: Path, records: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    task_name: str,
    prompt: str,
    marker: str,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    window_size: int = 0,
    max_samples: int = 0,
    sample_seed: int = 42,
) -> Dict[str, int]:
    marker_token = f"<{marker}>"
    records = build_sample_records(
        input_dir=input_dir,
        marker_token=marker_token,
        prompt=prompt,
        window_size=window_size,
    )
    records = maybe_subsample_records(records, max_samples=max_samples, sample_seed=sample_seed)
    train_records, valid_records, test_records = split_records(records, train_ratio=train_ratio, valid_ratio=valid_ratio, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / f"{task_name}_train.json", train_records)
    write_jsonl(output_dir / f"{task_name}_valid.json", valid_records)
    write_jsonl(output_dir / f"{task_name}_test.json", test_records)
    (output_dir / f"{task_name}_label.json").write_text(json.dumps(LABEL_MAP, indent=2), encoding="utf-8")

    return {
        "total_records": len(records),
        "train_records": len(train_records),
        "valid_records": len(valid_records),
        "test_records": len(test_records),
    }


def convert_uad_dataset(
    input_dir: Path,
    internal_output_dir: Path,
    export_output_dir: Path,
    dataset_name: str,
    task_name: str,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
    window_size: int = 0,
    max_samples: int = 0,
    sample_seed: int = 42,
    template_name: str = "uad_template_v1",
) -> Dict[str, int]:
    records = build_uad_records(
        input_dir=input_dir,
        dataset_name=dataset_name,
        template_name=template_name,
        window_size=window_size,
    )
    records = maybe_subsample_records(records, max_samples=max_samples, sample_seed=sample_seed)
    train_records, valid_records, test_records = split_records(records, train_ratio=train_ratio, valid_ratio=valid_ratio, seed=seed)

    split_map = {
        "train": train_records,
        "valid": valid_records,
        "test": test_records,
    }

    internal_output_dir.mkdir(parents=True, exist_ok=True)
    export_output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_records_list in split_map.items():
        internal_payload = []
        export_payload = []
        for split_record in split_records_list:
            payload = dict(split_record.record)
            payload["split"] = split_name
            internal_payload.append(payload)
            export_payload.append(export_uad_sample(payload))
        write_jsonl_dicts(internal_output_dir / f"{task_name}_{split_name}.json", internal_payload)
        write_jsonl_dicts(export_output_dir / f"{task_name}_{split_name}.json", export_payload)

    (export_output_dir / f"{task_name}_label.json").write_text(json.dumps(UAD_LABEL_MAP, indent=2), encoding="utf-8")

    return {
        "total_records": len(records),
        "train_records": len(train_records),
        "valid_records": len(valid_records),
        "test_records": len(test_records),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .data/.label pairs into TrafficLLM JSONL datasets.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing .data/.label pairs.")
    parser.add_argument("--output_dir", type=Path, help="Directory for generated legacy JSONL files.")
    parser.add_argument("--output_internal_dir", type=Path, help="Directory for generated internal UAD JSONL files.")
    parser.add_argument("--output_tllm_dir", type=Path, help="Directory for generated TrafficLLM-compatible UAD JSONL files.")
    parser.add_argument("--task_name", type=str, required=True, help="Prefix for generated dataset files.")
    parser.add_argument("--dataset_name", type=str, help="Dataset name for UAD exports.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Classification prompt appended to each instruction.")
    parser.add_argument("--marker", type=str, default=DEFAULT_MARKER, choices=["packet", "flow"], help="Traffic marker token.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--valid_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--window_size", type=int, default=0, help="Number of .data rows per sample window. Use 0 to keep one file as one sample.")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum number of samples to export after windowing. Use 0 for all samples.")
    parser.add_argument("--sample_seed", type=int, default=42, help="Shuffle seed for subsampling before split.")
    parser.add_argument("--template_name", type=str, default="uad_template_v1", choices=sorted(UAD_TEMPLATES), help="UAD instruction template name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_internal_dir or args.output_tllm_dir:
        if args.output_internal_dir is None or args.output_tllm_dir is None:
            raise ValueError("UAD conversion requires both --output_internal_dir and --output_tllm_dir.")
        if not args.dataset_name:
            raise ValueError("UAD conversion requires --dataset_name.")
        stats = convert_uad_dataset(
            input_dir=args.input_dir,
            internal_output_dir=args.output_internal_dir,
            export_output_dir=args.output_tllm_dir,
            dataset_name=args.dataset_name,
            task_name=args.task_name,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            window_size=args.window_size,
            max_samples=args.max_samples,
            sample_seed=args.sample_seed,
            template_name=args.template_name,
        )
    else:
        if args.output_dir is None:
            raise ValueError("Legacy conversion requires --output_dir.")
        stats = convert_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            task_name=args.task_name,
            prompt=args.prompt,
            marker=args.marker,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            window_size=args.window_size,
            max_samples=args.max_samples,
            sample_seed=args.sample_seed,
        )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
