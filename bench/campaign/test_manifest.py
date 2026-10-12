import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures as fx
import manifest as mf


class Lint(unittest.TestCase):
    def setUp(self):
        self.tmpd = tempfile.TemporaryDirectory()
        self.tmp = self.tmpd.name
        self.src, self.commit = fx.make_src(self.tmp)

    def tearDown(self):
        self.tmpd.cleanup()

    def errors(self, exps, **kw):
        m = mf.load(fx.make_manifest(self.tmp, self.commit, exps, **kw))
        return [str(f) for f in mf.lint(m, self.src) if f.level == "error"]

    def good(self):
        return [{"id": "probe.env", "kind": "probe", "priority": "P0", "est_min": 1, "probe": "env"},
                {"id": "cmp", "kind": "sweep", "priority": "P0", "est_min": 20, "file": "bench/sweep_x.txt", "outputs": ["sweep.jsonl"]},
                fx.py_exp("work", fx.WRITE_OK, est=5, needs=["probe.env"]),
                fx.py_exp("extra", fx.WRITE_OK, prio="P1", est=5)]

    def test_good_manifest_is_clean(self):
        self.assertEqual(self.errors(self.good()), [])

    def test_each_defect_is_an_error(self):
        def with_(**change):
            g = self.good()
            g[2].update(change)
            return g

        cases = {
            "placeholder": (with_(title="TBD later"), "placeholder"),
            "bad kind": (with_(kind="shell"), "kind"),
            "bad priority": (with_(priority="P9"), "priority"),
            "no estimate": (with_(est_min=0), "est_min"),
            "timeout below estimate": (with_(timeout_min=1), "timeout_min"),
            "unknown need": (with_(needs=["ghost"]), "unknown experiment"),
            "need on lower priority": (with_(needs=["extra"]), "lower priority"),
            "need listed later": (with_(needs=["extra"], priority="P1"), "listed after"),
            "no outputs": (with_(outputs=[]), "outputs"),
            "bad placeholder in argv": (with_(argv=["{nowhere}/x"]), "placeholder"),
            "duplicate id": (self.good() + [fx.py_exp("work", fx.WRITE_OK)], "duplicate"),
            "missing sweep file": ([dict(self.good()[1], file="bench/nothere.txt")] + self.good()[2:], "does not exist at the pinned commit"),
        }
        for name, (exps, want) in cases.items():
            with self.subTest(name):
                errs = self.errors(exps)
                self.assertTrue(any(want in e for e in errs), (name, errs))

    def test_budget(self):
        exps = self.good()
        exps[1]["est_min"] = 60                       # P0 sum 66 x 1.25 > 60
        self.assertTrue(any("exceeds the budget" in e for e in self.errors(exps, budget=60)))
        self.assertEqual(self.errors(exps, budget=100), [])
        self.assertTrue(any("exceeds wall_limit_min" in e for e in self.errors(exps, budget=200, wall=100)))

    def test_pins(self):
        m = mf.load(fx.make_manifest(self.tmp, "abc123", self.good()))
        errs = [str(f) for f in mf.lint(m, self.src) if f.level == "error"]
        self.assertTrue(any("40-hex" in e for e in errs))
        m = mf.load(fx.make_manifest(self.tmp, "f" * 40, self.good()))
        self.assertTrue(any("commit not found" in str(f) for f in mf.lint(m, self.src)))

    def test_submodule_and_gemmul8_pins_follow_the_pinned_commit(self):
        m = mf.load(fx.make_manifest(self.tmp, self.commit, self.good(), extra_campaign=f'gemmul8_commit = "{"a" * 40}"'))
        errs = [str(f) for f in mf.lint(m, self.src) if f.level == "error"]
        self.assertTrue(any("no bench/GEMMUL8_PIN" in e for e in errs), errs)
        fx.wr(os.path.join(self.src, "bench", "GEMMUL8_PIN"), f"# pin\ng8 {'a' * 40} 3\n")
        fx.git(self.src, "add", "-A")
        fx.git(self.src, "commit", "-qm", "pin")
        commit2 = fx.git(self.src, "rev-parse", "HEAD")
        m = mf.load(fx.make_manifest(self.tmp, commit2, self.good(), extra_campaign=f'gemmul8_commit = "{"a" * 40}"'))
        self.assertEqual([f for f in mf.lint(m, self.src) if f.level == "error"], [])
        m = mf.load(fx.make_manifest(self.tmp, commit2, self.good(), extra_campaign=f'gemmul8_commit = "{"b" * 40}"'))
        self.assertTrue(any("GEMMUL8_PIN" in str(f) for f in mf.lint(m, self.src) if f.level == "error"))
        m = mf.load(fx.make_manifest(self.tmp, commit2, self.good()))
        m.data["campaign"]["cutlass_commit"] = "d" * 40
        self.assertTrue(any("records CUTLASS" in str(f) for f in mf.lint(m, self.src) if f.level == "error"))

    def test_decisions_binding_and_content(self):
        path = fx.make_manifest(self.tmp, self.commit, self.good())
        fx.wr(os.path.join(self.tmp, "R0.decisions.toml"), fx.rd(os.path.join(self.tmp, "R0.decisions.toml")) + "\n# edited\n")
        errs = [str(f) for f in mf.lint(mf.load(path), self.src) if f.level == "error"]
        self.assertTrue(any("does not match the decisions file" in e for e in errs), errs)
        bad = {
            "unknown experiment": fx.DECISIONS.replace('exp = "{exp}"', 'exp = "ghost"'),
            "limits swapped": fx.DECISIONS.replace("nogo_below = 500.0", "nogo_below = 5000.0"),
            "unknown kind": fx.DECISIONS.replace('kind = "threshold"', 'kind = "oracle"'),
            "TBD limit": fx.DECISIONS.replace('field = "tops"', 'field = "TBD"'),
            "unknown parameter": fx.DECISIONS + 'colour = "red"\n',
            "missing limit": fx.DECISIONS.replace("go_at_least = 1000.0\n", ""),
        }
        for name, text in bad.items():
            with self.subTest(name):
                self.assertTrue(self.errors(self.good(), decisions=text), name)

    def test_classify_and_plan_order(self):
        m = mf.load(fx.make_manifest(self.tmp, self.commit, [
            fx.py_exp("c", fx.WRITE_OK, prio="P1"), fx.py_exp("b", fx.WRITE_OK, risky=True), fx.py_exp("a", fx.WRITE_OK),
            fx.py_exp("only-h", fx.WRITE_OK, classes=["other"])]))
        self.assertEqual(m.classify("NVIDIA FAKEGPU 9"), "fake")
        self.assertIsNone(m.classify("NVIDIA H100"))
        self.assertEqual([e["id"] for e in m.plan("fake")], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
