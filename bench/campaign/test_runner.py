import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures as fx
import results as rs

RUNNER = os.path.join(fx.HERE, "runner.py")
ENV = {**os.environ, "CAMPAIGN_GPU_NAME": "FAKEGPU 9000"}


PROCS = []


def run_runner(mpath, out, src, extra=(), env=None, wait=True, **kw):
    cmd = [fx.PY, "-I", RUNNER, "run", mpath, "--out", out, "--src", src, "--build", "/nonexistent", "--sync-min", "0.01", *extra]
    p = subprocess.Popen(cmd, env=env or ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)
    PROCS.append(p)
    if not wait:
        return p
    o, e = p.communicate(timeout=120)
    return p.returncode, o, e


def journal(out):
    return [json.loads(l) for l in fx.rd(os.path.join(out, "journal.jsonl")).splitlines()]


class RunnerBase(unittest.TestCase):
    def setUp(self):
        self.tmpd = tempfile.TemporaryDirectory()
        self.tmp = self.tmpd.name
        self.src, self.commit = fx.make_src(self.tmp)
        self.out = os.path.join(self.tmp, "out")

    def tearDown(self):
        while PROCS:
            p = PROCS.pop()
            p.poll() is None and p.kill()
            p.wait()
            p.stdout.close()
            p.stderr.close()
        self.tmpd.cleanup()

    def manifest(self, exps, **kw):
        return fx.make_manifest(self.tmp, self.commit, exps, **kw)


class Basics(RunnerBase):
    def test_run_to_completion(self):
        m = self.manifest([
            {"id": "probe.env", "kind": "probe", "priority": "P0", "est_min": 0.05, "probe": "env"},
            fx.py_exp("work", fx.WRITE_OK),
            fx.py_exp("bad", "import sys; open('out.jsonl','w').write('x'); sys.exit(3)", prio="P1"),
            fx.py_exp("nooutput", "pass", prio="P1"),
            fx.py_exp("dep", fx.WRITE_OK, prio="P2", needs=["bad"]),
        ])
        rc, o, e = run_runner(m, self.out, self.src)
        self.assertEqual(rc, 0, e)
        c = rs.read_json(os.path.join(self.out, rs.COMPLETE))
        self.assertEqual(c["states"], {"probe.env": "ok", "work": "ok", "bad": "failed", "nooutput": "failed"})
        self.assertIn("dep", c["skipped"])
        for eid in c["states"]:
            st, probs = rs.verify_exp(os.path.join(self.out, "exp", eid))
            self.assertEqual(st, c["states"][eid], probs)
        done = rs.read_json(os.path.join(self.out, "exp", "work", rs.DONE))
        self.assertIn("clocks_before", done)
        self.assertEqual(done["files"]["out.jsonl"]["bytes"], os.path.getsize(os.path.join(self.out, "exp", "work", "out.jsonl")))
        lock = rs.read_json(os.path.join(self.out, rs.LOCK))
        self.assertEqual((lock["src_commit"], lock["class"], lock["pin_check"]), (self.commit, "fake", "ok"))
        self.assertTrue(os.path.exists(os.path.join(self.out, "env.json")))

    def test_order_priority_then_risky_last(self):
        m = self.manifest([fx.py_exp("c-p1", fx.WRITE_OK, prio="P1"), fx.py_exp("b-risky", fx.WRITE_OK, risky=True),
                           fx.py_exp("a-p0", fx.WRITE_OK), fx.py_exp("d-p2", fx.WRITE_OK, prio="P2")])
        self.assertEqual(run_runner(m, self.out, self.src)[0], 0)
        order = [j["id"] for j in journal(self.out) if j["event"] == "exp-done"]
        self.assertEqual(order, ["a-p0", "b-risky", "c-p1", "d-p2"])

    def test_timeout_kills_the_process_group(self):
        m = self.manifest([fx.py_exp("slow", "import time; time.sleep(60)", est=0.02, timeout_min=0.03)])
        t0 = time.time()
        rc, o, e = run_runner(m, self.out, self.src)
        self.assertEqual(rc, 0, e)
        self.assertLess(time.time() - t0, 30)
        d = rs.read_json(os.path.join(self.out, "exp", "slow", rs.DONE))
        self.assertEqual(d["status"], "failed")
        self.assertIn("timeout", d["reason"])

    def test_not_enough_time_skips_but_p0_runs(self):
        m = self.manifest([fx.py_exp("p0", fx.WRITE_OK, est=0.05), fx.py_exp("p1", fx.WRITE_OK, prio="P1", est=5.0)])
        rc, o, e = run_runner(m, self.out, self.src, extra=["--wall-min", "7", "--reserve-min", "6"])
        self.assertEqual(rc, 0, e)
        c = rs.read_json(os.path.join(self.out, rs.COMPLETE))
        self.assertEqual(c["states"], {"p0": "ok"})
        self.assertIn("not enough time", c["skipped"]["p1"])

    def test_pin_check_refuses_other_commit_and_dirty_tree(self):
        m = self.manifest([fx.py_exp("work", fx.WRITE_OK)])
        fx.wr(os.path.join(self.src, "bench", "sweep_x.txt"), "edit\n", "a")
        rc, o, e = run_runner(m, self.out, self.src)
        self.assertEqual(rc, 3, e)
        fx.git(self.src, "commit", "-qam", "second")
        rc, o, e = run_runner(m, self.out, self.src)
        self.assertEqual(rc, 3, e)
        rc, o, e = run_runner(m, self.out, self.src, extra=["--no-pin-check"])
        self.assertEqual(rc, 0, e)
        self.assertEqual(rs.read_json(os.path.join(self.out, rs.LOCK))["pin_check"], "skipped")

    def test_relative_paths(self):
        m = self.manifest([fx.py_exp("work", fx.WRITE_OK)])
        cmd = [fx.PY, "-I", RUNNER, "run", os.path.relpath(m, self.tmp), "--out", "out", "--src", "src", "--build", "bld"]
        r = subprocess.run(cmd, cwd=self.tmp, env=ENV, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(rs.verify_exp(os.path.join(self.tmp, "out", "exp", "work"))[0], "ok")

    def test_lint_errors_block(self):
        m = self.manifest([fx.py_exp("work", fx.WRITE_OK, est=-1)])
        rc, o, e = run_runner(m, self.out, self.src)
        self.assertEqual(rc, 2, e)
        self.assertIn("est_min", e)


class Resume(RunnerBase):
    def counter_exp(self, eid, **kw):
        # appends one line to a counter file outside the result folder each time it really runs
        code = f"""
        import json
        open({os.path.join(self.tmp, eid + '.count')!r}, 'a').write('x')
        open('out.jsonl','w').write(json.dumps({{'method':'ceil','tops':1200.0,'status':'ok'}})+'\\n')
        """
        return fx.py_exp(eid, code, **kw)

    def test_finished_work_is_not_redone(self):
        m = self.manifest([self.counter_exp("work"), self.counter_exp("more", prio="P1")])
        self.assertEqual(run_runner(m, self.out, self.src)[0], 0)
        self.assertEqual(run_runner(m, self.out, self.src)[0], 0)
        for eid in ("work", "more"):
            self.assertEqual(fx.rd(os.path.join(self.tmp, eid + ".count")), "x")
        self.assertEqual([j["event"] for j in journal(self.out)].count("resume"), 1)
        self.assertTrue(os.path.exists(os.path.join(self.out, "env-resume-1.json")))

    def test_kill_9_then_resume(self):
        pidfile = os.path.join(self.tmp, "child.pid")
        flag = os.path.join(self.tmp, "release")
        hang = f"""
        import os, time
        open('partial.txt', 'w').write('half')
        if not os.path.exists({flag!r}):
            open({pidfile!r}, 'w').write(str(os.getpid()))
            time.sleep(120)
        open('partial.txt', 'w').write('whole')
        """
        m = self.manifest([self.counter_exp("first"), fx.py_exp("hang", hang, est=0.5, outputs=("partial.txt",)),
                           self.counter_exp("last", prio="P1")])
        p = run_runner(m, self.out, self.src, wait=False)
        for _ in range(200):
            if os.path.exists(pidfile):
                break
            time.sleep(0.05)
        self.assertTrue(os.path.exists(pidfile), "the hanging experiment never started")
        child = int(fx.rd(pidfile))
        p.send_signal(signal.SIGKILL)
        p.wait()
        time.sleep(0.5)                         # the child must die with the runner (PR_SET_PDEATHSIG)
        with self.assertRaises(ProcessLookupError):
            os.kill(child, 0)
        self.assertFalse(os.path.exists(os.path.join(self.out, rs.COMPLETE)))
        self.assertEqual(rs.verify_exp(os.path.join(self.out, "exp", "hang"))[0], "incomplete")
        self.assertEqual(rs.verify_exp(os.path.join(self.out, "exp", "first"))[0], "ok")
        fx.wr(flag, "")                 # the experiment can finish now
        rc, o, e = run_runner(m, self.out, self.src)
        self.assertEqual(rc, 0, e)
        self.assertEqual(fx.rd(os.path.join(self.tmp, "first.count")), "x")    # not redone
        self.assertEqual(fx.rd(os.path.join(self.tmp, "last.count")), "x")
        c = rs.read_json(os.path.join(self.out, rs.COMPLETE))
        self.assertEqual(c["states"], {"first": "ok", "hang": "ok", "last": "ok"})
        self.assertEqual(fx.rd(os.path.join(self.out, "exp", "hang", "partial.txt")), "whole")
        for eid in c["states"]:
            self.assertEqual(rs.verify_exp(os.path.join(self.out, "exp", eid))[0], "ok")

    def test_resume_refuses_another_manifest(self):
        m = self.manifest([self.counter_exp("work")])
        self.assertEqual(run_runner(m, self.out, self.src)[0], 0)
        m2 = self.manifest([self.counter_exp("work"), self.counter_exp("extra", prio="P1")])
        rc, o, e = run_runner(m2, self.out, self.src)
        self.assertEqual(rc, 4, e)
        self.assertIn("manifest_sha256", e)

    def test_failed_is_kept_unless_retry_failed(self):
        flag = os.path.join(self.tmp, "flag")
        code = f"""
        import os, json
        if not os.path.exists({flag!r}):
            raise SystemExit(5)
        open('out.jsonl','w').write(json.dumps({{'method':'ceil','tops':1200.0}})+'\\n')
        """
        m = self.manifest([fx.py_exp("flaky", code)])
        run_runner(m, self.out, self.src)
        self.assertEqual(rs.verify_exp(os.path.join(self.out, "exp", "flaky"))[0], "failed")
        fx.wr(flag, "")
        run_runner(m, self.out, self.src)
        self.assertEqual(rs.verify_exp(os.path.join(self.out, "exp", "flaky"))[0], "failed")
        run_runner(m, self.out, self.src, extra=["--retry-failed"])
        self.assertEqual(rs.verify_exp(os.path.join(self.out, "exp", "flaky"))[0], "ok")


class CopyBack(RunnerBase):
    def test_copy_back_after_each_experiment_and_complete_last(self):
        dest = os.path.join(self.tmp, "dest")
        m = self.manifest([fx.py_exp("work", fx.WRITE_OK)])
        rc, o, e = run_runner(m, self.out, self.src, extra=["--copy-to", dest])
        self.assertEqual(rc, 0, e)
        self.assertEqual(rs.verify_exp(os.path.join(dest, "exp", "work"))[0], "ok")
        self.assertTrue(os.path.exists(os.path.join(dest, rs.COMPLETE)))
        self.assertEqual(fx.rd(os.path.join(dest, "MANIFEST.toml")), fx.rd(m))

    def test_sigterm_syncs_and_exits_143(self):
        dest = os.path.join(self.tmp, "dest")
        pidfile = os.path.join(self.tmp, "child.pid")
        hang = f"import os,time; open({pidfile!r},'w').write(str(os.getpid())); time.sleep(120)"
        m = self.manifest([fx.py_exp("work", fx.WRITE_OK), fx.py_exp("hang", hang, est=0.5, outputs=("x",))])
        p = run_runner(m, self.out, self.src, extra=["--copy-to", dest], wait=False)
        for _ in range(200):
            if os.path.exists(pidfile):
                break
            time.sleep(0.05)
        p.send_signal(signal.SIGTERM)
        p.wait(timeout=60)
        self.assertEqual(p.returncode, 143)
        self.assertEqual(rs.verify_exp(os.path.join(dest, "exp", "work"))[0], "ok")
        self.assertFalse(os.path.exists(os.path.join(dest, rs.COMPLETE)))
        with self.assertRaises(ProcessLookupError):
            os.kill(int(fx.rd(pidfile)), 0)

    def test_sigusr1_syncs_while_running(self):
        dest = os.path.join(self.tmp, "dest")
        pidfile = os.path.join(self.tmp, "child.pid")
        hang = f"import os,time; open({pidfile!r},'w').write(str(os.getpid())); time.sleep(3)\nopen('x','w').write('x')"
        m = self.manifest([fx.py_exp("hang", hang, est=0.5, outputs=("x",))])
        p = run_runner(m, self.out, self.src, extra=["--copy-to", dest, "--sync-min", "100"], wait=False)
        for _ in range(200):
            if os.path.exists(pidfile):
                break
            time.sleep(0.05)
        p.send_signal(signal.SIGUSR1)
        for _ in range(100):
            if os.path.exists(os.path.join(dest, "journal.jsonl")) and "exp-start" in fx.rd(os.path.join(dest, "journal.jsonl")):
                break
            time.sleep(0.1)
        self.assertIn("exp-start", fx.rd(os.path.join(dest, "journal.jsonl")))
        self.assertFalse(os.path.exists(os.path.join(dest, rs.COMPLETE)))
        p.wait(timeout=60)


if __name__ == "__main__":
    unittest.main()
