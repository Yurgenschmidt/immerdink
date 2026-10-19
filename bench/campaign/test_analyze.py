import json
import os
import random
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an
import decisions as dc
import fixtures as fx
import results as rs
from test_runner import ENV, run_runner

GOLDEN = os.environ.get("CAMPAIGN_GOLDEN_DIR", os.path.expanduser("~/repos/generalWork/immerdink/scratch/runs/20261011-180837"))


class Folder(unittest.TestCase):
    """A real result folder made by the runner, then damaged in the ways the issue lists."""

    DECISIONS = fx.DECISIONS

    def setUp(self):
        self.tmpd = tempfile.TemporaryDirectory()
        self.tmp = self.tmpd.name
        self.src, self.commit = fx.make_src(self.tmp)
        self.out = os.path.join(self.tmp, "out")
        self.exps = [fx.py_exp("work", fx.WRITE_OK), fx.py_exp("other", fx.WRITE_OK, prio="P1")]
        self.m = fx.make_manifest(self.tmp, self.commit, self.exps, decisions=self.DECISIONS)
        self.dfile = os.path.join(self.tmp, "R0.decisions.toml")

    def tearDown(self):
        self.tmpd.cleanup()

    def run_clean(self, extra=()):
        rc, o, e = run_runner(self.m, self.out, self.src, extra=extra)
        self.assertEqual(rc, 0, e)

    def verdict(self, **kw):
        vs, info = an.verdicts(self.out, **kw)
        return {v["id"]: v for v in vs}

    def workfile(self, name="out.jsonl"):
        return os.path.join(self.out, "exp", "work", name)


class Integrity(Folder):
    def test_clean_folder_gives_a_verdict(self):
        self.run_clean()
        v = self.verdict(decisions_file=self.dfile)
        self.assertEqual(v["ceiling"]["verdict"], "GO", v)
        self.assertEqual(self.verdict()["ceiling"]["verdict"], "GO")

    def corrupt_cases(self):
        """name -> function(self) applied to a fresh clean folder; every one must give INVALID, never GO."""
        def truncate(s):
            p = s.workfile()
            data = fx.rd(p, "rb")
            fx.wr(p, data[: len(data) // 2], "wb")

        def same_size_edit(s):
            p = s.workfile()
            data = fx.rd(p).replace("1200.0", "9999.0")
            fx.wr(p, data)

        def delete_file(s):
            os.remove(s.workfile())

        def extra_file(s):
            fx.wr(s.workfile("surprise.txt"), "x")

        def lose_done(s):
            os.remove(s.workfile(rs.DONE))

        def edit_manifest_copy(s):
            fx.wr(os.path.join(s.out, "MANIFEST.toml"), "# edited\n", "a")

        def edit_decisions_copy(s):
            fx.wr(os.path.join(s.out, "DECISIONS.toml"), "# edited\n", "a")

        def lose_lock(s):
            os.remove(os.path.join(s.out, rs.LOCK))

        def wrong_commit_in_lock(s):
            p = os.path.join(s.out, rs.LOCK)
            d = rs.read_json(p)
            d["src_commit"] = "0" * 40
            rs.write_json_atomic(p, d)

        def dirty_in_lock(s):
            p = os.path.join(s.out, rs.LOCK)
            d = rs.read_json(p)
            d["src_dirty"] = True
            rs.write_json_atomic(p, d)

        return {"truncated": truncate, "same-size edit": same_size_edit, "deleted": delete_file, "unlisted file": extra_file,
                "DONE.json lost": lose_done, "manifest copy edited": edit_manifest_copy,
                "decisions copy edited": edit_decisions_copy, "LOCK lost": lose_lock, "wrong commit": wrong_commit_in_lock,
                "dirty tree": dirty_in_lock}

    def test_every_damage_is_invalid_and_never_go(self):
        for name, fn in self.corrupt_cases().items():
            with self.subTest(name):
                shutil.rmtree(self.out, ignore_errors=True)
                self.run_clean()
                fn(self)
                v = self.verdict(decisions_file=self.dfile)
                got = {x["verdict"] for x in v.values()}
                if name == "DONE.json lost":
                    self.assertEqual(got, {"INCONCLUSIVE"})          # looks like an experiment that never finished
                else:
                    self.assertEqual(got, {"INVALID"}, v)

    def test_edited_decisions_file_is_invalid(self):
        self.run_clean()
        fx.wr(self.dfile, fx.rd(self.dfile).replace("go_at_least = 1000.0", "go_at_least = 100.0"))
        v = self.verdict(decisions_file=self.dfile)
        self.assertEqual(v["ceiling"]["verdict"], "INVALID")
        self.assertIn("edited after the manifest", v["ceiling"]["reason"])

    def test_wrong_commit_run_is_invalid_unless_accepted(self):
        fx.wr(os.path.join(self.src, "bench", "sweep_x.txt"), "more\n", "a")
        fx.git(self.src, "commit", "-qam", "second")
        rc, o, e = run_runner(self.m, self.out, self.src, extra=["--no-pin-check"])
        self.assertEqual(rc, 0, e)
        self.assertEqual(self.verdict()["ceiling"]["verdict"], "INVALID")
        self.assertEqual(self.verdict(accept_unpinned=True)["ceiling"]["verdict"], "GO")

    def test_failed_experiment_is_inconclusive_not_invalid(self):
        self.exps[0] = fx.py_exp("work", "raise SystemExit(2)")
        self.m = fx.make_manifest(self.tmp, self.commit, self.exps, decisions=self.DECISIONS)
        self.run_clean()
        v = self.verdict()["ceiling"]
        self.assertEqual(v["verdict"], "INCONCLUSIVE")
        self.assertIn("work: failed", v["reason"])

    def test_wrong_card_class_is_invalid(self):
        self.run_clean()
        p = os.path.join(self.out, rs.LOCK)
        d = rs.read_json(p)
        d["gpu_name"] = "NVIDIA H100"
        rs.write_json_atomic(p, d)
        self.assertEqual(self.verdict()["ceiling"]["verdict"], "INVALID")

    def test_unfinished_job_is_reported(self):
        self.run_clean()
        os.remove(os.path.join(self.out, rs.COMPLETE))
        vs, info = an.verdicts(self.out)
        self.assertIn("did not finish", an.render(vs, info))


class Thresholds(Folder):
    def write_tops(self, value):
        code = f"import json; open('out.jsonl','w').write(json.dumps({{'method':'ceil','tops':{value},'status':'ok'}})+'\\n')"
        self.exps[0] = fx.py_exp("work", code)
        self.m = fx.make_manifest(self.tmp, self.commit, self.exps, decisions=self.DECISIONS)

    def test_bands(self):
        for value, want in ((1200.0, "GO"), (1000.0, "GO"), (750.0, "INCONCLUSIVE"), (499.0, "NO-GO")):
            with self.subTest(value):
                shutil.rmtree(self.out, ignore_errors=True)
                self.write_tops(value)
                self.run_clean()
                self.assertEqual(self.verdict()["ceiling"]["verdict"], want)


class FakeCtx:
    def __init__(self, recs, base=None):
        self.recs, self.base = recs, base or []
        self.base_dir = "."

    def jsonl(self, exp, file):
        return self.recs

    def base_jsonl(self, rel):
        return self.base


def duel_row(case, ratio, rng, n=300, sd=0.004, spread=1.0, bit=True, level=1e-3):
    ta = [level * (1.0 + rng.gauss(0.0, sd)) for _ in range(n)]
    tb = [x / ratio * (1.0 + rng.gauss(0.0, 0.0015)) for x in ta]
    return {"case": case, "method": "duel-b", "status": "ok", "pair_t_a": ta, "pair_t_b": tb, "spread_pct": spread, "bit_identical": bit}


class Decisions(unittest.TestCase):
    PAIRED = {"id": "tiles", "kind": "paired", "requires": ["k"], "exp": "k", "file": "duel.jsonl"}

    def test_paired_adopt_keep_inconclusive(self):
        rng = random.Random(11)
        v = dc.evaluate(self.PAIRED, FakeCtx([duel_row("a", 1.10, rng), duel_row("b", 1.0, rng)]))
        self.assertEqual(v["verdict"], "ADOPT", v)
        v = dc.evaluate(self.PAIRED, FakeCtx([duel_row("a", 1.0, rng), duel_row("b", 1.0, rng)]))
        self.assertEqual(v["verdict"], "KEEP", v)
        v = dc.evaluate(self.PAIRED, FakeCtx([duel_row("a", 1.10, rng), duel_row("b", 0.9, rng)]))
        self.assertEqual(v["verdict"], "INCONCLUSIVE", v)

    def test_paired_not_bit_identical_is_never_adopted(self):
        rng = random.Random(12)
        v = dc.evaluate(self.PAIRED, FakeCtx([duel_row("a", 1.2, rng, bit=False)]))
        self.assertEqual(v["verdict"], "KEEP")

    def test_paired_noisy_case_needs_ten_percent(self):
        rng = random.Random(13)
        v = dc.evaluate(self.PAIRED, FakeCtx([duel_row("a", 1.06, rng, spread=20.0)]))
        self.assertEqual(v["verdict"], "KEEP", v)
        v = dc.evaluate(self.PAIRED, FakeCtx([duel_row("a", 1.25, rng, spread=20.0)]))
        self.assertEqual(v["verdict"], "ADOPT", v)

    def test_paired_without_samples_is_inconclusive(self):
        v = dc.evaluate(self.PAIRED, FakeCtx([{"case": "a", "method": "duel-b", "status": "ok"}]))
        self.assertEqual(v["verdict"], "INCONCLUSIVE")

    GUARD = {"id": "g", "kind": "guard", "requires": ["k"], "exp": "k", "file": "x.jsonl", "baseline": "b.jsonl",
             "baseline_sha256": "0" * 64, "key": ["case", "method"], "field": "median_s"}

    def test_guard(self):
        base = [{"case": c, "method": "imd", "median_s": 1.0, "spread_pct": 1.0, "status": "ok"} for c in "abc"]
        new = [{"case": "a", "method": "imd", "median_s": 1.01, "spread_pct": 1.0, "status": "ok"},
               {"case": "b", "method": "imd", "median_s": 1.10, "spread_pct": 1.0, "status": "ok"},
               {"case": "c", "method": "imd", "median_s": 0.9, "spread_pct": 1.0, "status": "ok"}]
        v = dc.evaluate(self.GUARD, FakeCtx(new, base))
        self.assertEqual(v["verdict"], "FLAGGED")
        self.assertEqual([f["key"] for f in v["details"]["flagged"]], [["b", "imd"]])
        v = dc.evaluate(self.GUARD, FakeCtx(new[:1] + new[2:], base))
        self.assertEqual(v["verdict"], "PASS")


def cmp_rec(case, method, tf, N=14, err=1e-16, spread=1.0, **kw):
    r = {"case": case, "method": method, "N": N, "variant": "", "tflops_eq": tf, "median_s": 1.0 / tf, "spread_pct": spread,
         "max_normerr": err, "nonfinite": 0, "status": "ok", "m": 8, "n": 8, "k": 8, "b": 1}
    r.update(kw)
    return r


def parity_case(case, ratio, spread=1.0):
    """One case's comparison records: DGEMM at error 1e-15, immerdink (N=14) and GEMMul8 fast (N=14) below it."""
    return [cmp_rec(case, "dgemm", 1.0, err=1e-15), cmp_rec(case, "imd", ratio * 10.0, spread=spread),
            cmp_rec(case, "gemmul8", 10.0, spread=spread, g8_mode="fast")]


PARITY = {"id": "parity", "kind": "parity", "requires": ["c"], "exp": "c", "file": "compare.jsonl",
          "geomean_at_least": 1.0, "case_floor": 0.9, "exclude": ["s_batch"]}


class Parity(unittest.TestCase):
    def verdict(self, ratios, spread=1.0, dec=PARITY):
        recs = [r for c, x in ratios.items() for r in parity_case(c, x, spread)]
        return dc.evaluate(dec, FakeCtx(recs))

    def test_pass_fail_and_floor(self):
        v = self.verdict({"a": 1.2, "b": 1.0, "c": 1.0})
        self.assertEqual(v["verdict"], "PASS", v)
        v = self.verdict({"a": 1.3, "b": 1.3, "c": 0.85})              # geomean ok, one case below the 0.9 floor
        self.assertEqual(v["verdict"], "FAIL")
        self.assertEqual(v["details"]["below_floor"], ["c"])
        v = self.verdict({"a": 0.95, "b": 0.95})                       # geomean below 1
        self.assertEqual(v["verdict"], "FAIL")

    def test_spread_is_added_before_the_floor(self):
        v = self.verdict({"a": 1.5, "b": 1.5, "c": 0.85}, spread=10.0)  # 0.85 x 1.10 = 0.935
        self.assertEqual(v["verdict"], "PASS", v)

    def test_excluded_case_is_out_of_the_mean_but_reported(self):
        v = self.verdict({"a": 1.0, "b": 1.0, "s_batch": 70.0})
        self.assertAlmostEqual(v["details"]["geomean"], 1.0)
        self.assertGreater(v["details"]["geomean_all_cases"], 1.5)
        self.assertEqual(v["verdict"], "PASS")

    def test_missing_family_is_inconclusive(self):
        recs = parity_case("a", 1.1) + [r for r in parity_case("b", 1.1) if r["method"] != "gemmul8"]
        v = dc.evaluate(PARITY, FakeCtx(recs))
        self.assertEqual(v["verdict"], "INCONCLUSIVE")
        self.assertIn("b", v["reason"])

    def test_needs_dgemm_level_accuracy(self):
        recs = parity_case("a", 1.1)
        for r in recs:
            if r["method"] == "imd":
                r["max_normerr"] = 1e-9                                  # worse than DGEMM: not a DGEMM-level record
        self.assertEqual(dc.evaluate(PARITY, FakeCtx(recs))["verdict"], "INCONCLUSIVE")

    def test_xduel_records_feed_the_paired_decision(self):
        rng = random.Random(5)
        tb = [1e-3 * (1.0 + rng.gauss(0.0, 0.004)) for _ in range(200)]
        ti = [x / 1.3 for x in tb]
        rec = {"case": "a", "method": "xduel-imd", "status": "ok", "pair_t_a": tb, "pair_t_b": ti, "spread_pct": 1.0}
        d = {"id": "x", "kind": "paired", "requires": ["k"], "exp": "k", "file": "f", "method": "xduel-imd"}
        self.assertEqual(dc.evaluate(d, FakeCtx([rec]))["verdict"], "ADOPT")


try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


@unittest.skipUnless(HAVE_NUMPY, "the tile fit needs numpy (an environment without it gets INCONCLUSIVE, not an error)")
class TileRefit(unittest.TestCase):
    """The tile_refit decision (issue 20): bench/k4_fit.py on imd_bench-style records of a forced-tile sweep."""

    DEC = {"id": "k4-tiles", "kind": "tile_refit", "requires": ["tiles"], "exp": "tiles", "file": "sweep.jsonl"}

    @staticmethod
    def records(truth, sms=132, noise=0.01, seed=1):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        import k4_fit as K
        import test_k4_fit as T
        out = []
        for o in T.synth(truth, sms, noise, seed):
            out.append({"case": o.case, "N": o.N, "k4_config": o.cfg, "sm_count": sms, "status": "ok",
                        "phase_view": f"{o.m}x{o.nh}x{o.kh} b={o.b} n_hat={o.nh} k_hat={o.kh}",
                        "phases": {"K4 gemm": {"ms": o.ms}}})
        return out

    def test_shipped_constants_keep(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import k4_fit as K
        v = dc.evaluate(self.DEC, FakeCtx(self.records(K.SHIPPED_TABLE)))
        self.assertEqual(v["verdict"], "KEEP", v)

    def test_a_card_with_cheap_small_tiles_adopts(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import k4_fit as K
        truth = K.Table([(2716, (418,)), (1500, (300, 300)), (700, (350, 400)), (500, (500, 560, 450))])
        v = dc.evaluate(self.DEC, FakeCtx(self.records(truth)))
        self.assertEqual(v["verdict"], "ADOPT", v)
        self.assertIn("table", v["details"])

    def test_two_cards_in_one_file_are_inconclusive(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import k4_fit as K
        recs = self.records(K.SHIPPED_TABLE)
        recs[0]["sm_count"] = 188
        self.assertEqual(dc.evaluate(self.DEC, FakeCtx(recs))["verdict"], "INCONCLUSIVE")
SHAPES = {  # case: (m=n=k, K4 ms, P ms, T* ms) from the H100 run of issue 11
    "sq4096": (4096, 2.02, 1.23, 2.23), "sq8192": (8192, 15.5, 4.57, 14.9)}


def rp_inputs(stock, lt=None, mma=1200.0, s=0.99, exact=True, spread=1.0, g8_spread=1.0, drop=(), i32=None):
    """Records for the rate_projection decision from the H100 numbers; stock/lt are TOPS per case."""
    sweep, xd, ce = [], [], []
    for case, (d, k4, pk, ts) in SHAPES.items():
        if case in drop:
            continue
        sweep.append({"case": case, "method": "imd", "N": 14, "status": "ok", "m": d, "n": d, "k": d, "b": 1, "k_hat": d,
                      "spread_pct": spread, "phases": {"K4 gemm": {"ms": k4}, "KL A": {"ms": pk / 2}, "K5 crt": {"ms": pk / 2}}})
        xd.append({"case": case, "method": "gemmul8", "g8_mode": "fast", "N": 14, "status": "ok", "median_s": ts * 1e-3, "spread_pct": g8_spread})
        xd.append({"case": case, "method": "xduel-imd", "N": 14, "status": "ok", "median_s": s * (k4 + pk) * 1e-3, "spread_pct": spread})
        ce.append({"variant": "hopper:128x256x128:1x2x1:coop", "status": "ok", "m": d, "n": d, "k": d, "batch": 14, "out": "i8",
                   "tops": stock[case], "exact": exact, "spread_pct": spread})
        if i32 is not False:
            ce.append({"variant": "hopper:128x256x128:1x2x1:coop", "status": "ok", "m": d, "n": d, "k": d, "batch": 14, "out": "i32",
                       "tops": stock[case] * 0.9 if i32 is None else i32[case], "exact": exact, "spread_pct": spread})
        if lt:
            ce.append({"variant": "cublaslt", "status": "ok", "m": d, "n": d, "k": d, "batch": 1, "out": "i32", "tops": lt[case],
                       "exact": True, "spread_pct": 1.0})
    ce.append({"variant": "mma_peak", "status": "ok", "tops": mma, "exact": None})
    return sweep, xd, ce


class RpCtx:
    def __init__(self, sweep, xd, ce):
        self.d = {"s.jsonl": sweep, "x.jsonl": xd, "c.jsonl": ce}
        self.base_dir = "."

    def jsonl(self, exp, file):
        return self.d[file]


RP = {"id": "wgmma", "kind": "rate_projection", "requires": ["s", "x", "c"], "exp": "s", "file": "s.jsonl", "xduel_exp": "x",
      "xduel_file": "x.jsonl", "ceiling_exp": "c", "ceiling_file": "c.jsonl", "cases": ["sq4096", "sq8192"], "gain_go": 1.3,
      "pack_factor_go": 0.67, "pack_factor_opt": 0.5, "window": 1.05, "mma_margin": 0.9, "n": 14, "g8_n": 14, "s_min": 0.95, "s_max": 1.08,
      "u_floor": 0.05, "u_noisy": 0.10, "noisy_spread_pct": 5.0}


def lint_rp(dec):
    """Lint findings (errors) of a one-decision file."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "d.toml")
        with open(path, "w") as f:
            f.write("[[decision]]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in dec.items()))
        return [x for x in dc.lint(path, {"s": {}, "x": {}, "c": {}}) if x.level == "error"]


class RateProjection(unittest.TestCase):
    def verdict(self, *a, dec=RP, **kw):
        return dc.evaluate(dec, RpCtx(*rp_inputs(*a, **kw)))

    def test_go_inconclusive_nogo(self):
        v = self.verdict({"sq4096": 2000.0, "sq8192": 2000.0})
        self.assertEqual(v["verdict"], "GO", v)
        self.assertAlmostEqual(v["details"]["cases"][0]["r_k4"], 952, delta=5)
        v = self.verdict({"sq4096": 1100.0, "sq8192": 1100.0})             # gain 1.15 / 1.11: below the GO gain, optimistic case holds
        self.assertEqual(v["verdict"], "INCONCLUSIVE", v)
        v = self.verdict({"sq4096": 800.0, "sq8192": 800.0}, lt={"sq4096": 850.0, "sq8192": 850.0})
        self.assertEqual(v["verdict"], "NO-GO", v)

    def test_projection_not_gain_decides_at_4096(self):
        v = self.verdict({"sq4096": 1300.0, "sq8192": 2000.0})            # gain 1.37 at 4096 but the 5 % window is missed (B)
        c = v["details"]["cases"][0]
        self.assertTrue(c["A"] and not c["B"], c)
        self.assertEqual(v["verdict"], "INCONCLUSIVE")

    def test_mma_peak_high_means_tuning_may_suffice(self):
        v = self.verdict({"sq4096": 2000.0, "sq8192": 2000.0}, mma=1900.0)  # mma.sync alone could reach the needed rate: C fails
        self.assertEqual(v["verdict"], "INCONCLUSIVE", v)
        self.assertFalse(v["details"]["cases"][0]["C"])

    def test_noise_haircut(self):
        v = self.verdict({"sq4096": 1500.0, "sq8192": 1500.0}, spread=1.0)
        noisy = self.verdict({"sq4096": 1500.0, "sq8192": 1500.0}, spread=30.0)
        self.assertLess(v["details"]["cases"][0]["u"], noisy["details"]["cases"][0]["u"])
        self.assertGreaterEqual(noisy["details"]["cases"][0]["u"], 0.30)

    def test_invalid_inputs_never_go(self):
        self.assertEqual(self.verdict({"sq4096": 2000.0, "sq8192": 2000.0}, exact=False)["verdict"], "INCONCLUSIVE")   # no exact stock rate
        self.assertEqual(self.verdict({"sq4096": 2000.0, "sq8192": 2000.0}, s=1.3)["verdict"], "INCONCLUSIVE")        # anchor off
        self.assertEqual(self.verdict({"sq4096": 2000.0, "sq8192": 2000.0}, drop=("sq8192",))["verdict"], "INCONCLUSIVE")
        # a missing shape does not hide a NO-GO at the other one
        v = self.verdict({"sq4096": 700.0, "sq8192": 700.0}, lt={"sq4096": 700.0, "sq8192": 700.0}, drop=("sq8192",))
        self.assertEqual(v["verdict"], "NO-GO", v)

    def test_registered_cases_only(self):
        for cases in ([], ["sq4096"], ["sq4096", "sq8192", "sq1024"]):
            v = self.verdict({"sq4096": 2000.0, "sq8192": 2000.0}, dec=dict(RP, cases=cases))
            self.assertEqual(v["verdict"], "INCONCLUSIVE", (cases, v))
        self.assertTrue(any("registered" in str(f) for f in lint_rp(dict(RP, cases=[]))))

    def test_huge_spread_is_not_a_pass(self):
        # a spread of 100 % or more must not turn the haircut into a negative rate (and a GO)
        v = self.verdict({"sq4096": 1500.0, "sq8192": 1500.0}, spread=130.0)
        self.assertEqual(v["verdict"], "INCONCLUSIVE", v)
        v = self.verdict({"sq4096": 2000.0, "sq8192": 2000.0}, g8_spread=100.0)
        self.assertEqual(v["verdict"], "INCONCLUSIVE", v)

    def test_missing_spread_is_not_zero(self):
        sweep, xd, ce = rp_inputs({"sq4096": 2000.0, "sq8192": 2000.0})
        for r in sweep:
            r.pop("spread_pct")
        self.assertEqual(dc.evaluate(RP, RpCtx(sweep, xd, ce))["verdict"], "INCONCLUSIVE")

    def test_int8_rate_needs_an_exact_int32_twin(self):
        # int8 only (no int32 run of the variant): the saturated int8 check alone is not trusted
        v = self.verdict({"sq4096": 2000.0, "sq8192": 2000.0}, i32=False)
        self.assertEqual(v["verdict"], "INCONCLUSIVE", v)
        # the optimistic case uses the best exact stock rate of any output type: a fast int32 record prevents a false NO-GO
        v = self.verdict({"sq4096": 800.0, "sq8192": 800.0}, i32={"sq4096": 2000.0, "sq8192": 2000.0})
        self.assertNotEqual(v["verdict"], "NO-GO", v)

    def test_xduel_imd_is_the_one_after_its_baseline(self):
        sweep, xd, ce = rp_inputs({"sq4096": 2000.0, "sq8192": 2000.0})
        other = dict(next(r for r in xd if r["method"] == "gemmul8" and r["case"] == "sq4096"), g8_mode="exact", median_s=1.0)
        stray = dict(next(r for r in xd if r["method"] == "xduel-imd" and r["case"] == "sq4096"), median_s=9.0)
        xd2 = [other, stray] + xd                      # a stray record of another pairing in front of the right pair
        self.assertEqual(dc.evaluate(RP, RpCtx(sweep, xd2, ce))["verdict"], "GO")

    def test_registered_parameters_are_required(self):
        d = {k: v for k, v in RP.items() if k != "s_max"}
        self.assertTrue(any("missing s_max" in str(f) for f in lint_rp(d)))

    def test_confirm_role(self):
        d = dict(RP, role="confirm", confirm_gain=1.2)
        self.assertEqual(self.verdict({"sq4096": 1300.0, "sq8192": 1300.0}, dec=d)["verdict"], "PASS")
        self.assertEqual(self.verdict({"sq4096": 1000.0, "sq8192": 1300.0}, dec=d)["verdict"], "FAIL")
        self.assertEqual(self.verdict({"sq4096": 1300.0, "sq8192": 1300.0}, drop=("sq8192",), dec=d)["verdict"], "INCONCLUSIVE")


@unittest.skipUnless(os.path.isdir(GOLDEN), "the issue-11 result folders are not on this machine")
class ParityGolden(unittest.TestCase):
    """The parity gate on the issue-11 comparison records gives the numbers of the tracking issue: geometric mean over 16
    cases (s_batch out) 0.84 on H100 and 0.78 on H200, 1.09 and 1.01 with s_batch in; s_batch 70x; Blackwell passes."""

    def run_gate(self, suffix):
        job = [d for d in os.listdir(GOLDEN) if d.endswith(suffix)][0]
        recs = rs.read_jsonl(os.path.join(GOLDEN, job, "compare.jsonl"))
        return dc.evaluate(PARITY, FakeCtx(recs))["details"], dc.evaluate(PARITY, FakeCtx(recs))

    def test_h100_h200_blackwell(self):
        d, v = self.run_gate("h100")
        self.assertEqual(len([c for c in d["cases"] if not c["excluded"]]), 16)
        self.assertAlmostEqual(d["geomean"], 0.84, delta=0.005)
        self.assertAlmostEqual(d["geomean_all_cases"], 1.09, delta=0.005)
        self.assertAlmostEqual(next(c["ratio"] for c in d["cases"] if c["case"] == "s_batch"), 70.0, delta=1.0)
        self.assertEqual(v["verdict"], "FAIL")
        d, v = self.run_gate("h200")
        self.assertAlmostEqual(d["geomean"], 0.78, delta=0.005)
        self.assertAlmostEqual(d["geomean_all_cases"], 1.01, delta=0.005)
        self.assertEqual(self.run_gate("blackwell")[1]["verdict"], "PASS")


class GuardWithSamples(unittest.TestCase):
    """A record that carries its repetitions is judged by the robust spread (a range of 12 % would hide a 10 % slowdown)."""

    def test_guard_decision_uses_samples(self):
        rng = random.Random(21)

        def rec(level):
            xs = [level * (1.0 + rng.gauss(0.0, 0.004)) for _ in range(60)]
            xs[3] *= 1.12                                          # one outlier: range spread ~ 13 %
            return {"case": "a", "method": "imd", "median_s": sorted(xs)[30], "spread_pct": 13.0, "samples_s": xs, "status": "ok"}

        g = dict(Decisions.GUARD)
        v = dc.evaluate(g, FakeCtx([rec(1.10e-3)], [rec(1.0e-3)]))
        self.assertEqual(v["verdict"], "FLAGGED", v)
        v = dc.evaluate(g, FakeCtx([rec(1.0e-3)], [rec(1.0e-3)]))
        self.assertEqual(v["verdict"], "PASS", v)
        no_samples = lambda r: {k: x for k, x in r.items() if k != "samples_s"}
        v = dc.evaluate(g, FakeCtx([no_samples(rec(1.10e-3))], [no_samples(rec(1.0e-3))]))
        self.assertEqual(v["verdict"], "PASS", v)                  # the summary alone cannot see it: the range is 13 %


@unittest.skipUnless(os.path.isdir(GOLDEN), "the issue-11 result folders are not on this machine")
class Golden(unittest.TestCase):
    """The analyzer regenerates the issue-11 tables (summary, energy, cuTENSOR workspace, comparison, comparison energy) from the
    job folders of that submission and gets the same text as the jobs wrote on the cluster."""

    def jobs(self):
        return sorted(os.path.join(GOLDEN, d) for d in os.listdir(GOLDEN) if "-immerdink-" in d)

    def test_tables_are_reproduced(self):
        jobs = self.jobs()
        self.assertEqual(len(jobs), 3)
        for j in jobs:
            checked, diffs = an.golden(j)
            self.assertGreaterEqual(len(checked), 4, j)
            self.assertEqual(diffs, [], j)

    def test_a_changed_record_is_seen(self):
        with tempfile.TemporaryDirectory() as t:
            j = os.path.join(t, "job")
            os.makedirs(j)
            src = self.jobs()[0]
            for n in ("compare.jsonl", "compare.md"):
                shutil.copy(os.path.join(src, n), j)
            lines = fx.rd(os.path.join(j, "compare.jsonl")).splitlines()
            for i, l in enumerate(lines):
                r = json.loads(l)
                if r.get("method") == "imd" and r.get("case") == "sq4096" and r.get("N") == 14:
                    r["median_s"] *= 1.5
                    r["tflops_eq"] /= 1.5
                    lines[i] = json.dumps(r)
            fx.wr(os.path.join(j, "compare.jsonl"), "\n".join(lines) + "\n")
            checked, diffs = an.golden(j)
            self.assertTrue(diffs)


if __name__ == "__main__":
    unittest.main()
