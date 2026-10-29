import copy
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archgate as ag

K = {"regs": 100, "stack": 0, "spill_st": 0, "spill_ld": 0}


def table():
    return {"sm_89": {"src/k4/k4_cfg0.cu": {"kernels": {"_Zk": dict(K)}, "sass": {"IMMA": 8}}},
            "sm_90a": {"src/k4/k4_cfg0.cu": {"kernels": {"_Zk": dict(K)}, "sass": {"IMMA": 8}}}}


class Compare(unittest.TestCase):
    RULES = {"must_contain": {"sm_89": {"src/k4/k4_cfg0.cu": ["IMMA"]}},
             "must_not_contain": {"sm_89": ["IGMMA"]}}

    def diffs(self, mutate):
        new = copy.deepcopy(table())
        mutate(new)
        return ag.compare(new, {"table": table()}, self.RULES)

    def test_identical(self):
        self.assertEqual(self.diffs(lambda t: None), [])

    def test_each_change_is_reported(self):
        cases = {
            "register change": (lambda t: t["sm_89"]["src/k4/k4_cfg0.cu"]["kernels"]["_Zk"].update(regs=101), "changed"),
            "new spill": (lambda t: t["sm_90a"]["src/k4/k4_cfg0.cu"]["kernels"]["_Zk"].update(spill_st=16, regs=100), "SPILLS"),
            "new kernel": (lambda t: t["sm_89"]["src/k4/k4_cfg0.cu"]["kernels"].update(_Zn=dict(K)), "new kernel"),
            "kernel gone": (lambda t: t["sm_89"]["src/k4/k4_cfg0.cu"]["kernels"].clear(), "kernel gone"),
            "sass change": (lambda t: t["sm_89"]["src/k4/k4_cfg0.cu"]["sass"].update(IMMA=16), "families changed"),
            "unknown source": (lambda t: t["sm_89"].update({"src/new.cu": {"kernels": {}, "sass": {}}}), "not in the committed table"),
        }
        for name, (fn, want) in cases.items():
            with self.subTest(name):
                self.assertTrue(any(want in d for d in self.diffs(fn)), self.diffs(fn))

    def test_rules(self):
        d = self.diffs(lambda t: t["sm_89"]["src/k4/k4_cfg0.cu"]["sass"].update(IGMMA=2))
        self.assertTrue(any("must not contain IGMMA" in x for x in d), d)
        d = self.diffs(lambda t: t["sm_89"]["src/k4/k4_cfg0.cu"]["sass"].pop("IMMA"))
        self.assertTrue(any("must contain IMMA" in x for x in d), d)

    def test_committed_table_is_well_formed(self):
        t = json.load(open(ag.DEFAULT_TABLE))
        self.assertEqual(set(t["table"]), set(ag.DEFAULT_ARCHS))
        for arch, files in t["table"].items():
            self.assertIn("src/k4/k4_cfg0.cu", files)
            self.assertTrue(files["src/k4/k4_cfg0.cu"]["sass"].get("IMMA"), arch)
            for rel, d in files.items():
                if rel.startswith("src/k4/"):       # the K4 kernels must not spill on any architecture
                    for k, v in d["kernels"].items():
                        self.assertEqual((v.get("stack", 0), v.get("spill_st", 0), v.get("spill_ld", 0)), (0, 0, 0), (arch, rel, k))


if __name__ == "__main__":
    unittest.main()
