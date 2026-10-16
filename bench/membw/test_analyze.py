import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an

DEV = {"kind": "membw_device", "gpu": "X", "sm_count": 100, "l2_bytes": 50 << 20, "peak_gbs": 1000.0}


def mb(pattern, gbs, bps=1, th=256, ilp=4):
    return {"kind": "membw", "pattern": pattern, "blocks_per_sm": bps, "threads": th, "ilp": ilp, "gbs": gbs, "pct_of_peak": gbs / 10.0}


def rec(ms, case="sq4096", name="KL A", m=4096, k=4096, N=14):
    return {"case": case, "method": "imd", "status": "ok", "m": m, "n": 4096, "k": k, "b": 1, "N": N, "k_hat": k, "phases": {name: {"ms": ms}}}


class Judge(unittest.TestCase):
    MEMBW = [DEV, mb("fan", 800.0), mb("fan", 900.0, 4, 512), mb("fanin", 880.0)]

    def verdict(self, bench):
        return an.judge(self.MEMBW, bench)[3]

    def test_bytes(self):
        b = an.phase_bytes(rec(1), "KL A")
        self.assertEqual(b, 8 * 4096 * 4096 + 14 * 4096 * 4096 + 3 * 4096)
        self.assertEqual(an.phase_bytes(rec(1), "K5 crt"), 14 * 4096 * 4096 + 8 * 4096 * 4096)
        self.assertIsNone(an.phase_bytes(rec(1), "K4 gemm"))

    def test_bandwidth_bound(self):
        by = an.phase_bytes(rec(1), "KL A")
        ms = by / (880.0 * 1e6)             # 880 GB/s: 98% of the best fan (900)
        self.assertEqual(self.verdict([rec(ms)])[0]["verdict"], "BANDWIDTH-BOUND")

    def test_shape_bound(self):
        by = an.phase_bytes(rec(1), "KL A")
        ms = by / (300.0 * 1e6)             # 300 GB/s while the card does 900
        r = self.verdict([rec(ms)])[0]
        self.assertEqual(r["verdict"], "SHAPE-BOUND")
        self.assertAlmostEqual(r["pct_of_best"], 100 * 300 / 900, places=3)

    def test_small_phase_is_not_judged(self):
        self.assertEqual(self.verdict([rec(0.01, m=256, k=256)])[0]["verdict"], "SMALL")

    def test_crt_uses_fan_in(self):
        by = an.phase_bytes(rec(1), "K5 crt")
        r = self.verdict([rec(by / (870.0 * 1e6), name="K5 crt")])[0]
        self.assertEqual((r["pattern"], r["verdict"]), ("fanin", "BANDWIDTH-BOUND"))


if __name__ == "__main__":
    unittest.main()
