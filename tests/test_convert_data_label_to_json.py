import json
import tempfile
import unittest
from pathlib import Path

from scripts.convert_data_label_to_json import build_sample_records, convert_dataset


class ConvertDataLabelToJsonTest(unittest.TestCase):
    def test_build_sample_records_aggregates_binary_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "normal.data").write_text("pkt_a\npkt_b\n", encoding="utf-8")
            (tmp_path / "normal.label").write_text("00", encoding="utf-8")
            (tmp_path / "attack.data").write_text("pkt_c\npkt_d\n", encoding="utf-8")
            (tmp_path / "attack.label").write_text("01", encoding="utf-8")

            records = build_sample_records(tmp_path, "<packet>", "Classify benign or attack.")

            self.assertEqual(len(records), 2)
            by_name = {record.sample_name: record for record in records}
            self.assertEqual(by_name["normal"].output, "benign")
            self.assertEqual(by_name["attack"].output, "attack")
            self.assertIn("<packet>:", by_name["normal"].instruction)
            self.assertIn("pkt_a pkt_b", by_name["normal"].instruction)

    def test_build_sample_records_splits_large_file_into_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "windowed.data").write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")
            (tmp_path / "windowed.label").write_text("00101", encoding="utf-8")

            records = build_sample_records(
                tmp_path,
                "<packet>",
                "Classify benign or attack.",
                window_size=2,
            )

            self.assertEqual(len(records), 3)
            self.assertEqual([record.sample_name for record in records], ["windowed__w0000", "windowed__w0001", "windowed__w0002"])
            self.assertEqual([record.output for record in records], ["benign", "attack", "attack"])
            self.assertIn("l1 l2", records[0].instruction)
            self.assertIn("l3 l4", records[1].instruction)
            self.assertIn("l5", records[2].instruction)

    def test_convert_dataset_writes_jsonl_splits_and_label_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            input_dir.mkdir()

            samples = {
                "sample1": ("a1\na2\n", "00"),
                "sample2": ("b1\nb2\n", "01"),
                "sample3": ("c1\nc2\n", "00"),
                "sample4": ("d1\nd2\n", "11"),
            }
            for name, (data_text, label_text) in samples.items():
                (input_dir / f"{name}.data").write_text(data_text, encoding="utf-8")
                (input_dir / f"{name}.label").write_text(label_text, encoding="utf-8")

            stats = convert_dataset(
                input_dir=input_dir,
                output_dir=output_dir,
                task_name="demo",
                prompt="Classify benign or attack.",
                marker="packet",
                train_ratio=0.5,
                valid_ratio=0.25,
                seed=7,
                window_size=2,
            )

            self.assertEqual(stats["total_records"], 4)
            self.assertEqual(stats["train_records"], 2)
            self.assertEqual(stats["valid_records"], 1)
            self.assertEqual(stats["test_records"], 1)

            train_lines = (output_dir / "demo_train.json").read_text(encoding="utf-8").strip().splitlines()
            valid_lines = (output_dir / "demo_valid.json").read_text(encoding="utf-8").strip().splitlines()
            test_lines = (output_dir / "demo_test.json").read_text(encoding="utf-8").strip().splitlines()
            label_map = json.loads((output_dir / "demo_label.json").read_text(encoding="utf-8"))

            self.assertEqual(len(train_lines), 2)
            self.assertEqual(len(valid_lines), 1)
            self.assertEqual(len(test_lines), 1)
            self.assertEqual(label_map, {"benign": 0, "attack": 1})

            record = json.loads(train_lines[0])
            self.assertEqual(set(record.keys()), {"instruction", "output"})
            self.assertTrue(record["instruction"].startswith("Given the following traffic data <packet>:"))
            self.assertIn(record["output"], {"benign", "attack"})

    def test_convert_dataset_limits_total_records_with_max_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            input_dir.mkdir()

            for idx in range(6):
                (input_dir / f"sample{idx}.data").write_text(f"x{idx}\ny{idx}\n", encoding="utf-8")
                (input_dir / f"sample{idx}.label").write_text("01", encoding="utf-8")

            stats = convert_dataset(
                input_dir=input_dir,
                output_dir=output_dir,
                task_name="subset",
                prompt="Classify benign or attack.",
                marker="packet",
                train_ratio=0.5,
                valid_ratio=0.25,
                seed=7,
                window_size=2,
                max_samples=3,
                sample_seed=9,
            )

            total_lines = 0
            for split in ["train", "valid", "test"]:
                path = output_dir / f"subset_{split}.json"
                total_lines += len(path.read_text(encoding="utf-8").strip().splitlines()) if path.read_text(encoding="utf-8").strip() else 0

            self.assertEqual(stats["total_records"], 3)
            self.assertEqual(total_lines, 3)


if __name__ == "__main__":
    unittest.main()
