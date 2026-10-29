import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import random
import unittest

import stats


def series(rng, n, level=1.0, sd=0.004):
    return [level * (1.0 + rng.gauss(0.0, sd)) for _ in range(n)]


class MedianCI(unittest.TestCase):
    def test_too_few(self):
        self.assertIsNone(stats.median_ci([1, 2, 3, 4, 5]))
        self.assertIsNotNone(stats.median_ci([1, 2, 3, 4, 5, 6]))

    def test_coverage(self):
        rng = random.Random(7)
        hit = 0
        trials = 600
        for _ in range(trials):
            xs = [rng.gauss(0.0, 1.0) for _ in range(41)]
            lo, hi = stats.median_ci(xs)
            hit += lo <= 0.0 <= hi
        self.assertGreaterEqual(hit / trials, 0.93)     # nominal 0.95, conservative

    def test_large_n_agrees_with_exact_scale(self):
        rng = random.Random(3)
        xs = [rng.gauss(10.0, 1.0) for _ in range(3000)]
        lo, hi = stats.median_ci(xs)
        self.assertTrue(lo < stats.median(xs) < hi)
        self.assertLess(hi - lo, 0.15)


class Paired(unittest.TestCase):
    def test_same_series_is_a_tie(self):
        rng = random.Random(1)
        ta = series(rng, 200)
        tb = [x * (1.0 + rng.gauss(0.0, 0.001)) for x in ta]
        p = stats.paired_ratio(ta, tb)
        self.assertEqual(stats.classify_pair(p), "tie")

    def test_real_gain(self):
        rng = random.Random(2)
        ta = series(rng, 200)
        tb = [x / 1.08 * (1.0 + rng.gauss(0.0, 0.002)) for x in ta]
        p = stats.paired_ratio(ta, tb)
        self.assertEqual(stats.classify_pair(p, spread=2.0), "faster")
        self.assertAlmostEqual(p.ratio, 1.08, places=2)
        self.assertEqual(stats.classify_pair(stats.paired_ratio(tb, ta), spread=2.0), "slower")

    def test_noisy_case_needs_ten_percent(self):
        rng = random.Random(5)
        ta = series(rng, 300, sd=0.02)
        tb = [x / 1.06 * (1.0 + rng.gauss(0.0, 0.001)) for x in ta]
        p = stats.paired_ratio(ta, tb)
        self.assertEqual(stats.classify_pair(p, spread=2.0), "faster")       # quiet case: 6 % counts
        self.assertEqual(stats.classify_pair(p, spread=20.0), "tie")         # noisy case: under 10 % is a tie
        tc = [x / 1.25 for x in ta]
        self.assertEqual(stats.classify_pair(stats.paired_ratio(ta, tc), spread=20.0), "faster")

    def test_few_pairs_are_inconclusive(self):
        p = stats.paired_ratio([1.0, 1.1, 1.0], [0.5, 0.5, 0.5])
        self.assertEqual(stats.classify_pair(p), "inconclusive")

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            stats.paired_ratio([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            stats.paired_ratio([1.0, 0.0], [1.0, 1.0])


class Guard(unittest.TestCase):
    """Issue 15 'done when': the guard ignores halves of the same data and flags a 10 % slowdown."""

    def quiet(self, seed, n=400):
        rng = random.Random(seed)
        return series(rng, n, level=0.002, sd=0.004)

    def test_halves_of_the_same_data_are_not_flagged(self):
        for seed in range(25):
            xs = self.quiet(seed)
            flagged, slowdown, tol = stats.guard_samples(xs[::2], xs[1::2])
            self.assertFalse(flagged, (seed, slowdown, tol))
            flagged, slowdown, tol = stats.guard_samples(xs[: len(xs) // 2], xs[len(xs) // 2:])
            self.assertFalse(flagged, (seed, slowdown, tol))

    def test_ten_percent_slowdown_is_flagged(self):
        for seed in range(25):
            xs = self.quiet(seed)
            ys = [x * 1.10 for x in self.quiet(seed + 1000)]
            flagged, slowdown, tol = stats.guard_samples(xs, ys)
            self.assertTrue(flagged, (seed, slowdown, tol))
            self.assertAlmostEqual(slowdown, 0.10, delta=0.02)

    def test_real_rtx4090_repetitions(self):
        """Real per-repetition times of immerdink on the RTX 4090 (testdata/): their range spread is 8-12 %, the robust
        spread about 1 %, so halves of one series pass the guard and a 10 % slowdown does not slip through."""
        import json
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "real_samples_4090.json")) as f:
            cases = json.load(f)["cases"]
        self.assertGreaterEqual(len(cases), 3)
        for name, c in cases.items():
            x = c["samples_s"]
            with self.subTest(name):
                self.assertGreater(stats.spread_pct(x), 5.0)             # the range alone would hide a 10 % slowdown
                self.assertLess(stats.robust_spread_pct(x), 3.0)
                self.assertFalse(stats.guard_samples(x[::2], x[1::2])[0])
                self.assertFalse(stats.guard_samples(x[: len(x) // 2], x[len(x) // 2:])[0])
                self.assertTrue(stats.guard_samples(x, [v * 1.10 for v in x])[0])

    def test_floor_and_spread(self):
        self.assertFalse(stats.guard(1.0, 0.5, 1.015, 0.5)[0])          # inside the 2 % floor
        self.assertTrue(stats.guard(1.0, 0.5, 1.03, 0.5)[0])
        self.assertFalse(stats.guard(1.0, 12.0, 1.20, 3.0)[0])          # inside 2 x the larger spread
        self.assertTrue(stats.guard(1.0, 12.0, 1.30, 3.0)[0])
        self.assertFalse(stats.guard(1.0, 0.5, 0.80, 0.5)[0])           # a speed-up is not a regression


if __name__ == "__main__":
    unittest.main()
