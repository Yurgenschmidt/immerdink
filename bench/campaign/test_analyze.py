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
