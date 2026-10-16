import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import tempfile
import unittest

import summarize_tests as st

JUNIT = """<?xml version="1.0"?>
<testsuite name="x" tests="4" failures="1" disabled="0" skipped="1">
 <testcase name="a" status="run"/>
 <testcase name="b" status="run"><failure message="boom"/></testcase>
 <testcase name="c" status="notrun"><skipped/></testcase>
 <testcase name="d" status="run"/>
</testsuite>"""


class Summ(unittest.TestCase):
    def write(self, d, name, text):
        p = os.path.join(d, name)
        with open(p, "w") as f:
            f.write(text)
        return p

    def test_counts_and_names(self):
        with tempfile.TemporaryDirectory() as d:
            j = self.write(d, "a.xml", JUNIT)
            r = st.summarize([j], None, 0)
            self.assertEqual((r["tests_total"], r["tests_failed"], r["tests_skipped"]), (4, 1, 1))
            self.assertEqual(r["failed"], ["b", "c (skipped)"])
            self.assertAlmostEqual(r["pass_fraction"], 0.5)

    def test_lost_tests_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            j = self.write(d, "a.xml", '<testsuite><testcase name="a"/><testcase name="b"/></testsuite>')
            self.assertEqual(st.summarize([j], None, 0)["pass_fraction"], 1.0)
            r = st.summarize([j], None, 4)
            self.assertAlmostEqual(r["pass_fraction"], 0.5)
            self.assertIn("expected", r["note"])

    def test_two_junit_files_add_up(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.write(d, "a.xml", '<testsuite><testcase name="a"/></testsuite>')
            b = self.write(d, "b.xml", '<testsuite><testcase name="b"/><testcase name="c"/></testsuite>')
            self.assertEqual(st.summarize([a, b], None, 0)["tests_total"], 3)

    def test_hashes_last_line_of_a_case_wins(self):
        with tempfile.TemporaryDirectory() as d:
            j = self.write(d, "a.xml", '<testsuite><testcase name="a"/></testsuite>')
            h = self.write(d, "h.jsonl", "\n".join(json.dumps(x) for x in (
                {"name": "x", "match": False}, {"name": "x", "match": True}, {"name": "y", "match": False})) + "\n")
            r = st.summarize([j], h, 0)
            self.assertEqual((r["hash_cases"], r["hash_mismatch"], r["hash_mismatched"]), (2, 1, ["y"]))
            self.assertAlmostEqual(r["hash_match_fraction"], 0.5)

    def test_damaged_input_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(st.summarize([os.path.join(d, "none.xml")], None, 0)["status"], "error")
            j = self.write(d, "bad.xml", "<testsuite><testcase")
            self.assertEqual(st.summarize([j], None, 0)["status"], "error")
            g = self.write(d, "g.xml", '<testsuite><testcase name="a"/></testsuite>')
            h = self.write(d, "h.jsonl", "not json\n")
            self.assertEqual(st.summarize([g], h, 0)["status"], "error")

    def test_main_appends_one_line(self):
        with tempfile.TemporaryDirectory() as d:
            j = self.write(d, "a.xml", '<testsuite><testcase name="a"/></testsuite>')
            out = os.path.join(d, "o.jsonl")
            self.assertEqual(st.main([out, "--junit", j]), 0)
            self.assertEqual(st.main([out, "--junit", j]), 0)
            with open(out) as f:
                self.assertEqual(len(f.readlines()), 2)


if __name__ == "__main__":
    unittest.main()
