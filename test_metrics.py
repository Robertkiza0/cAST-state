import unittest

from metrics import compute_em, compute_es, compute_pass_at_1


class TestComputeEM(unittest.TestCase):
    def test_exact_match_returns_one(self):
        self.assertEqual(compute_em("return x + 1", "return x + 1"), 1)

    def test_mismatch_returns_zero(self):
        self.assertEqual(compute_em("return x + 1", "return x - 1"), 0)

    def test_whitespace_only_differences_are_ignored(self):
        self.assertEqual(compute_em("  return x + 1  ", "return x + 1"), 1)

    def test_extra_trailing_lines_in_prediction_are_ignored(self):
        self.assertEqual(compute_em("return x + 1", "return x + 1\nsome_other_line()"), 1)

    def test_empty_target_never_matches(self):
        self.assertEqual(compute_em("", ""), 0)


class TestComputeES(unittest.TestCase):
    def test_identical_strings_score_one(self):
        self.assertEqual(compute_es("return x + 1", "return x + 1"), 1.0)

    def test_both_empty_scores_one(self):
        self.assertEqual(compute_es("", ""), 1.0)

    def test_completely_different_strings_score_low(self):
        self.assertLess(compute_es("aaaa", "zzzz"), 0.5)

    def test_score_is_bounded_between_zero_and_one(self):
        score = compute_es("return compute_stats(values)", "print('totally unrelated')")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_partial_overlap_scores_between_bounds(self):
        score = compute_es("return x + 1", "return x + 2")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class TestComputePassAt1(unittest.TestCase):
    def test_matches_exact_match_exactly(self):
        """compute_pass_at_1 est délibérément un alias de compute_em pour ces
        datasets line-level sans harnais d'exécution — voir metrics.py."""
        pairs = [("return x + 1", "return x + 1"), ("return x + 1", "return x - 1"), ("", "")]
        for target, prediction in pairs:
            self.assertEqual(compute_pass_at_1(target, prediction), compute_em(target, prediction))


if __name__ == "__main__":
    unittest.main()
