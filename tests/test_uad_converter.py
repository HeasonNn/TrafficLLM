import json
import tempfile
import unittest
from pathlib import Path

from scripts.convert_data_label_to_json import (
    UAD_TEMPLATE_V1,
    build_uad_record,
    convert_uad_dataset,
    export_uad_sample,
)


class UadConverterTest(unittest.TestCase):
    def test_build_uad_record_with_flow_metadata(self):
        record = build_uad_record(
            dataset="DoHBrw",
            split="train",
            flow_id="flow-1",
            window_id=0,
            tokens=["aa", "bb"],
            window_label_text="01",
            flow_label_text="0010",
            template_name="uad_template_v1",
            window_start=0,
            window_end=2,
            sequence_length=4,
        )

        self.assertEqual(record["task"], "UAD")
        self.assertEqual(record["dataset"], "DoHBrw")
        self.assertEqual(record["split"], "train")
        self.assertEqual(record["flow_id"], "flow-1")
        self.assertEqual(record["window_id"], 0)
        self.assertEqual(record["instruction"], UAD_TEMPLATE_V1)
        self.assertEqual(record["input"], "aa bb")
        self.assertEqual(record["label"], "abnormal")
        self.assertEqual(record["flow_label"], "abnormal")
        self.assertEqual(record["meta"]["window_start"], 0)
        self.assertEqual(record["meta"]["window_end"], 2)
        self.assertEqual(record["meta"]["window_size"], 2)
        self.assertEqual(record["meta"]["original_sequence_length"], 4)
        self.assertEqual(record["meta"]["abnormal_count_in_window"], 1)
        self.assertEqual(record["meta"]["abnormal_ratio_in_window"], 0.5)

    def test_export_uad_sample_to_trafficllm_format(self):
        exported = export_uad_sample({
            "instruction": UAD_TEMPLATE_V1,
            "input": "aa bb",
            "label": "normal",
        })

        self.assertEqual(
            exported,
            {
                "instruction": UAD_TEMPLATE_V1,
                "input": "aa bb",
                "output": "normal",
            },
        )

    def test_convert_uad_dataset_writes_internal_and_export_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            internal_dir = tmp_path / "internal"
            export_dir = tmp_path / "tllm"
            input_dir.mkdir()

            samples = {
                "sample1": ("a1\na2\n", "00"),
                "sample2": ("b1\nb2\n", "01"),
                "sample3": ("c1\nc2\n", "10"),
                "sample4": ("d1\nd2\n", "00"),
            }
            for name, (data_text, label_text) in samples.items():
                (input_dir / f"{name}.data").write_text(data_text, encoding="utf-8")
                (input_dir / f"{name}.label").write_text(label_text, encoding="utf-8")

            stats = convert_uad_dataset(
                input_dir=input_dir,
                internal_output_dir=internal_dir,
                export_output_dir=export_dir,
                dataset_name="DoHBrw",
                task_name="uad_demo",
                train_ratio=0.5,
                valid_ratio=0.25,
                seed=7,
                window_size=2,
            )

            self.assertEqual(stats["total_records"], 4)
            self.assertEqual(stats["train_records"], 2)
            self.assertEqual(stats["valid_records"], 1)
            self.assertEqual(stats["test_records"], 1)

            internal_train = [json.loads(line) for line in (internal_dir / "uad_demo_train.json").read_text(encoding="utf-8").splitlines() if line.strip()]
            export_train = [json.loads(line) for line in (export_dir / "uad_demo_train.json").read_text(encoding="utf-8").splitlines() if line.strip()]
            label_map = json.loads((export_dir / "uad_demo_label.json").read_text(encoding="utf-8"))

            self.assertEqual(set(internal_train[0].keys()), {"task", "dataset", "split", "flow_id", "window_id", "instruction", "input", "label", "flow_label", "meta"})
            self.assertEqual(set(export_train[0].keys()), {"instruction", "input", "output"})
            self.assertEqual(label_map, {"normal": 0, "abnormal": 1})


if __name__ == "__main__":
    unittest.main()
