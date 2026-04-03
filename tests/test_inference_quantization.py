import unittest

from inference import build_model_load_kwargs, resolve_quantization_mode, resolve_task_code


class InferenceQuantizationTest(unittest.TestCase):
    def test_resolve_quantization_mode_prefers_explicit_flag(self):
        self.assertEqual(resolve_quantization_mode({"load_in_4bit": True, "load_in_8bit": True}), "4bit")
        self.assertEqual(resolve_quantization_mode({"load_in_8bit": True}), "8bit")
        self.assertEqual(resolve_quantization_mode({}), None)

    def test_build_model_load_kwargs_enables_low_cpu_mem_usage(self):
        kwargs = build_model_load_kwargs({"load_in_4bit": True})
        self.assertTrue(kwargs["low_cpu_mem_usage"])
        self.assertTrue(kwargs["load_in_4bit"])
        self.assertNotIn("torch_dtype", kwargs)

        kwargs = build_model_load_kwargs({"load_in_8bit": True})
        self.assertTrue(kwargs["load_in_8bit"])
        self.assertTrue(kwargs["low_cpu_mem_usage"])

        kwargs = build_model_load_kwargs({})
        self.assertEqual(kwargs["torch_dtype"], "auto")
        self.assertTrue(kwargs["low_cpu_mem_usage"])

    def test_resolve_task_code_uses_aliases_before_failing(self):
        config = {
            "tasks": {"Encrypted VPN Detection": "EVD"},
            "task_aliases": {"Website Fingerprinting": "EVD"},
        }
        self.assertEqual(resolve_task_code(config, "Encrypted VPN Detection"), "EVD")
        self.assertEqual(resolve_task_code(config, "Website Fingerprinting"), "EVD")
        with self.assertRaises(KeyError):
            resolve_task_code(config, "Unknown Task")


if __name__ == "__main__":
    unittest.main()
