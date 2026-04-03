import unittest

from inference import compute_window_abnormal_score, parse_uad_output


class UadInferenceTest(unittest.TestCase):
    def test_parse_uad_output_accepts_normal_and_abnormal_only(self):
        self.assertEqual(parse_uad_output("normal"), "normal")
        self.assertEqual(parse_uad_output(" AbNormal \n"), "abnormal")
        self.assertIsNone(parse_uad_output("maybe"))

    def test_compute_window_abnormal_score_prefers_abnormal_token(self):
        score = compute_window_abnormal_score(score_normal=-2.0, score_abnormal=-0.2)
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_compute_window_abnormal_score_prefers_normal_token(self):
        score = compute_window_abnormal_score(score_normal=-0.1, score_abnormal=-3.0)
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 0.5)


if __name__ == "__main__":
    unittest.main()
