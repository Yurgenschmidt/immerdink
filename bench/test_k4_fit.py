"""Self-test of bench/k4_fit.py (issue 20). CPU only; python3 -I bench/test_k4_fit.py"""
import math
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json

import k4_fit as K

A3 = os.path.expanduser("~/repos/immerdink-eval/legacy/bench/results")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def synth(table, sms, noise, seed, shapes=None, Ns=(8, 14)):
    """Observations a card with the cost table `table` would give."""
    rng = random.Random(seed)
    shapes = shapes or SHAPES
    out = []
    for name, (m, nh, kh, b) in shapes.items():
        for N in Ns:
            for c in range(4):
                t = K.estimate_fs(table, c, m, nh, kh, N * b, sms) * 1e-12     # fs -> ms... fs = 1e-15 s; ms = 1e-3 s
                ms = K.estimate_fs(table, c, m, nh, kh, N * b, sms) * 1e-12
                ms *= 1.0 + rng.gauss(0.0, noise)
                out.append(K.Obs(name, N, c, ms, m, nh, kh, b))
    return out


def _shapes():
    d = {}
    rng = random.Random(7)
    for i in range(48):
        m = rng.choice([64, 128, 256, 384, 512, 768, 1024, 2048, 4096, 8192])
        nh = rng.choice([64, 128, 256, 512, 1024, 2048, 4096])
        kh = rng.choice([128, 256, 512, 1024, 4096, 16384])
        b = rng.choice([1, 1, 1, 4, 16])
        d[f"c{i:02d}"] = (m, nh, kh, b)
    return d


SHAPES = _shapes()


class Model(unittest.TestCase):
    def test_pick_matches_the_library(self):
        """k4_select (libimmerdink_host, imd_lowering k4_config for a given SM count) against the Python mirror."""
        lib = os.path.join(ROOT, "build")
        sys.path.insert(0, os.path.join(ROOT, "python"))
        os.environ.setdefault("IMMERDINK_LIB_DIR", lib)
        try:
            from immerdink import host
            host.load()
        except Exception as e:                   # no library built here
            self.skipTest(f"host library not available: {e}")
        rng = random.Random(3)
        n = 0
        for _ in range(150):
            m, nh, kh = rng.choice([64, 100, 128, 500, 1000, 4096]), 16 * rng.randint(1, 600), 64 * rng.randint(1, 300)
            N, b, sms = rng.choice([8, 14]), rng.choice([1, 3, 16]), rng.choice([46, 128, 132, 188])
            lw = host.lowering("ikl,jkl->ijl", {"i": m, "j": nh, "k": kh, "l": b}, N=N, g=0, sm_count=sms)
            if lw.dispatch != "EMULATE":
                continue
            want = lw.k4_config
            got = K.pick(K.SHIPPED_TABLE, lw.m, lw.n_hat, lw.k_hat, N * lw.b, sms)
            self.assertEqual(got, want, (m, nh, kh, N, b, sms))
            n += 1
        self.assertGreater(n, 50)


class Fit(unittest.TestCase):
    def test_recovers_a_known_table(self):
        t = K.Table([(3100, (390,)), (2900, (430, 470)), (1700, (520, 590)), (1500, (900, 1000, 780))])
        obs = synth(t, 132, 0.01, 1)
        got, rel = K.fit_table(obs, 132)
        for (r0, m0), (r1, m1) in zip(t.costs, got.costs):
            self.assertLess(abs(r1 / r0 - 1), 0.15)
            for a, b in zip(m0, m1):
                self.assertLess(abs(b / a - 1), 0.08)

    def test_a3_data_reproduces_the_shipped_table(self):
        n, p = os.path.join(A3, "a3-configs-nsys.jsonl"), os.path.join(A3, "a3-configs-phases.jsonl")
        if not (os.path.exists(n) and os.path.exists(p)):
            self.skipTest("the A3 raw data (immerdink-eval/legacy) is not on this machine")
        obs = [o for o in K.load_records([n, p]) if o.case != "s_df"]
        t, _ = K.fit_table(obs, 128)
        self.assertEqual(t.costs, K.SHIPPED_TABLE.costs)

    def test_nonnegative(self):
        obs = synth(K.SHIPPED_TABLE, 128, 0.2, 5)
        t, _ = K.fit_table(obs, 128)
        for r, m in t.costs:
            self.assertGreaterEqual(r, 0)
            self.assertTrue(all(x >= 0 for x in m))


class Decide(unittest.TestCase):
    def test_same_card_keeps_the_table(self):
        obs = synth(K.SHIPPED_TABLE, 128, 0.01, 2)
        d = K.decide(obs, 128)
        self.assertEqual(d["verdict"], "KEEP", d["reason"])

    def test_a_different_card_adopts_when_it_helps(self):
        # a card whose small tiles are far cheaper than the shipped table believes: the old table never picks them
        truth = K.Table([(2716, (418,)), (1500, (300, 300)), (700, (350, 400)), (500, (500, 560, 450))])
        shapes = {k: v for k, v in SHAPES.items()}
        obs = synth(truth, 132, 0.01, 4, shapes)
        d = K.decide(obs, 132)
        self.assertEqual(d["verdict"], "ADOPT", d["reason"])
        self.assertGreater(d["gain"], K.GAIN_MIN)

    def test_noise_keeps_the_table(self):
        # the shipped table is right; 25% noise must not make a refit look better by luck on the held-out cases
        obs = synth(K.SHIPPED_TABLE, 128, 0.25, 6)
        d = K.decide(obs, 128)
        self.assertEqual(d["verdict"], "KEEP", d["reason"])

    def test_too_few_cases_is_inconclusive(self):
        obs = synth(K.SHIPPED_TABLE, 128, 0.01, 2, dict(list(SHAPES.items())[:5]))
        self.assertEqual(K.decide(obs, 128)["verdict"], "INCONCLUSIVE")

    def test_holdout_is_every_fourth_by_name(self):
        self.assertEqual(K.holdout_cases(["d", "a", "c", "b", "h", "e", "g", "f"]), {"d", "h"})


class Occupancy(unittest.TestCase):
    def test_the_4090_matches_the_shipped_assumption(self):
        self.assertEqual([K.occupancy("rtx4090", c) for c in range(4)], [c["ctas"] for c in K.CONFIGS])

    def test_hopper_holds_more_small_tiles(self):
        self.assertGreater(K.occupancy("h100", 3), K.CONFIGS[3]["ctas"])

    def test_smem_bytes(self):
        self.assertEqual([K.smem_bytes(c) for c in range(4)], [73728, 49152, 49152, 32768])   # NOTES-A3: 72, 48, 48, 32 KB


class Identify(unittest.TestCase):
    def test_a_constant_no_shape_uses_is_infinite(self):
        # one tile per modulus: the 64x64 tile with 3 CTAs per SM never has a 2- or 3-CTA round
        sd = K.identify([(64, 64, 64, 1)], 128, Ns=(8,))
        self.assertTrue(any(math.isinf(x) for x in sd[3]))

    def test_many_shapes_pin_every_constant_down(self):
        sd = K.identify(list(SHAPES.values()), 132)
        for row in sd:
            self.assertTrue(all(x < 0.10 for x in row), sd)


if __name__ == "__main__":
    unittest.main()
