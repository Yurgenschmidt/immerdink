#!/usr/bin/env python3
"""NOTES-C1 §4.3 M-d2: precision schedules timed on the GPU against their own [M] prediction
band (pre-registered protocol; DESIGN v1.19 §6g.9-§6g.10).

    python3 bench/accuracy/md2_time.py predict [--out PRED.json]                 # CPU only
    QUIET_GATE=1 tools/run_quiet.sh python3 bench/accuracy/md2_time.py run SYSTEM \\
        [--data normal|decay_opp] --out RECORDS.jsonl [--md2-cleared REASON]        # GPU
    python3 bench/accuracy/md2_time.py judge RECORDS.jsonl [--out SUMMARY.json]   # CPU only
    python3 bench/accuracy/md2_time.py selftest     # CPU only, standard library: the rules

ORDERING (binding, NOTES-C1 §5.9 "M-d2 ordering", §5.10 MD2-0). M-d2 runs only after M-c's
MC-3 path closes (MC-3a's recorded outcome, or Eric's explicit decision). `run` refuses the
M-d2 systems (ppl, df, sq4096, jacobi) unless --md2-cleared cites the commit (an ancestor of
the run's own) and the NOTES-C1 section ("NOTES-C1 §n.m") that record MC-3's closure
(recorded in every record); the tooling checks `smoke` (512^3) and `smoke_t` (a small
PPL-like tensor case), not M-d2 systems, run without it. If the calibration changes (MC-3a),
the bars, the band classes and REGISTERED_CALIB below are re-derived with `predict` and
committed BEFORE any M-d2 run (IR23), and this driver is re-registered (NOTES-C1 §4.3 names
its commit and sha256); `judge` refuses records whose calibration is not the registered one.

RECORDS, FREEZE AND CONTINUATION (binding, NOTES-C1 §4.3, §5.10 MD2-8 as amended):
- One normative records file: immerdink-eval results/c1/m_d2/md2.jsonl. Every M-d2 record
  `run` ever writes, in any file, is an attempt under rules (b) and (c) below and is
  reported; starting another file resets nothing.
- Until the first M-d2 record exists, re-registering this driver is a commit plus NOTES-C1
  §4.3 and the findings naming it. From the first M-d2 record on, the driver is FROZEN: it
  is re-registered only with an independent-review verdict recorded in NOTES-C1 §5.10.
  Records made under an earlier driver stay reported and binding, judged by the driver that
  made them (`git show <commit>:bench/accuracy/md2_time.py`, run in a checkout of <commit>).
- After rule (d) (two inadmissible attempts) or M-d2 STOPPED, nothing continues
  automatically. A further attempt, a new records file, or a restart after a library fix
  needs a recorded reviewer verdict or Eric's decision (NOTES-C1) that names the cause or
  the fix and the new file; every earlier record stays reported.
- A records file `judge` refuses (a provenance failure, below) is a procedural breach, not
  a reset: it is recorded in NOTES-C1 (which record, how it got there), and continuing in a
  new file needs a recorded reviewer verdict or Eric's decision.
- The library: `run` takes the M-d2 systems from ROOT/build/libimmerdink.so only, built at
  the registered commit before the first M-d2 run (the findings name its sha256).

The protocol (pre-registered before any M-d2 measurement):
- Systems (NOTES-C1 §4.3): the o = 10, v = 96 PPL view ijcd,abcd->ijab (GEMM view
  100 x 9216 x 9216; invariant operand B, the integrals), DF acQ,bdQ->acbd 64^4, Q = 2032
  (invariant B), 4096^3 ik,kj->ij (invariant A; the squares' spec of the cited [M] figures,
  NOTES-C1 §5.9 MC-4), all GATED; the Jacobi view abcd,cdij->abij (1600 x 100 x 1600,
  invariant A, the matrix W) reported, not gated.
- Configuration per level: PrecisionPlan.configs() over levels (6, 8, 9, 10, 14), in-line,
  GENERATE, RN (M-b's view of the §2.4 schedules; it picks g = N at every level on these
  shapes). The packed runs pack the invariant operand with the in-line plan's own
  pack_spec (bit-identical to the in-line call; checked per level before timing).
- Runs (one rep = each once, in an order rotated by the rep index):
    S14 = 5 x N6 + 4 x N8 + 4 x N10 + 3 x N14 (NOTES-C1 §2.4), F14 = 16 x N14   [gated pair]
    S9  = 6 x N6 + 10 x N9, F9 = 16 x N9, F8 = 16 x N8                           [reported]
    P14 = S14 with the invariant operand packed into ONE buffer of the largest pack_bytes,
          packed before each phase (the initial pack and the three re-packs at the
          switches), PF14 = one pack at N = 14, then 16 packed calls                [reported]
  Every call: alpha = 1, beta = 0, the same operands (an iteration's operands are fixed in
  shape; their values do not change the GEMM work), one CUDA stream, the raw entry points
  Plan.contract_ptr / Pack.pack_ptr (the checked calls run in the bit-exact check).
  The bit-exact check (rule (a)) fills C with a NaN bit pattern (NAN_BITS) before the in-line
  reference call and again before the packed call, and counts a level as bit-exact only if
  the in-line call wrote every element (no pattern left) and the packed result equals it
  bit for bit (so a packed call that writes nothing cannot compare equal to a reference
  that is itself vacuous). Registered with this fix at the commit NOTES-C1 §4.3 names.
- Timing: GPU events. Before each run a cudaMemsetAsync of a scratch buffer (--primer-mib,
  default 1024) keeps the GPU busy while the host enqueues the run, so the run's calls
  follow each other on the GPU without host gaps (queued_ahead records whether the whole
  run was enqueued before the primer finished); then an event before the first call and one
  after each call and each pack. A run's time is first event to last; per-call and per-pack
  times are consecutive differences.
- Conditions: one fresh process per (system, data), under the quiet gate, the GPU lock and
  memcap (tools/run_quiet.sh with QUIET_GATE=1), FREE only (ehrenfest busy check before
  each chunk, chunks <= 5 min); stock clocks, no clock lock (DESIGN §6c.5's regime, as M-c);
  a warm-up of >= 3 s of the system's own in-line N = 14 call (bench/warm_run.sh's 3 s),
  then one call at every level, in-line and packed, before the timed reps; --reps 15;
  if the worst spread (max - min) / median of the S14 and F14 totals exceeds 5 % (warm_run's
  BENCH_MAX_SPREAD), the reps are repeated once (2 attempts, M-c's BENCH_RETRIES) and the
  attempt with the smaller worst spread is kept. Clock state (nvidia-smi) after the warm-up
  and after the reps.
- Data: --data normal ((u - 1/2) exp(0.5 g), the bench default's distribution, seed 1) and
  --data decay_opp (bench/accuracy/c2_fixtures.py's family on the GEMM view, K in the spec's
  mode order; [KT26]'s data-dependent INT8 throughput). Both gated.
- The gated statistic: r = median over reps of the S14 totals / median of the F14 totals.
  Band: r_bar x (1 -/+ p90), p90 = DESIGN §6c.5's 90th-percentile target of the call class
  (>= 1 ms: 15 %; every level of every gated system is predicted >= 1 ms), and r <= 0.8.
  A system whose levels' predicted (DEFAULT/COLD) calls span classes takes the largest p90
  among them (the Jacobi view: < 100 us at N = 6, 8, 100 us - 1 ms at 9-14, so 30 %).
  r_bar = the registered bar (SYSTEMS' "bar": 0.750 / 0.664 / 0.673, NOTES-C1 §4.3, M-b;
  [M] shipped calibration, DEFAULT regime, COLD, GENERATE, g = N). The Jacobi view's band
  (predicted 0.786, < 100 µs class, p90 30 %) is reported only.
- Admissibility, re-runs and the decision (pre-registered with the reviewer's verdict,
  NOTES-C1 §5.10 MD2-3, MD2-6; `selftest` checks them). Per (system, data):
  (a) a record is INADMISSIBLE, neither PASS nor FAIL, when contaminated is not false (the
      kept attempt's worst S14/F14 spread > 5 %) or bitexact_packed_vs_inline is not true at
      every level. A bit-exact failure (false at any level) on any record, an M-d2 system's
      or a smoke's, is a library defect: it STOPS M-d2 (judge: M-d2 STOPPED; NOTES-C1 §4.3:
      on any record made with the M-d2 library, in any file) until the defect is fixed and
      recorded.
  (b) At most one re-run, in a fresh process after a FREE busy check in its own call,
      appending to the same records file. `run` refuses a re-run once an admissible record
      exists, a third record, and any M-d2 system once the file holds a bit-exact failure.
  (c) The first admissible record in file order decides: INSIDE or OUTSIDE the band. Every
      record is reported; a later record never replaces the first, and a record past the
      second is reported as excess, never considered.
  (d) If both attempts are inadmissible, the pair is INCOMPLETE, and M-d2 is not decided
      until the cause is recorded (NOTES-C1); nothing continues automatically (above).
  A gated system PASSES when its normal and decay_opp pairs are both INSIDE, FAILS when
  either is OUTSIDE, else is INCOMPLETE. M-d2: STOPPED on any bit-exact failure in the file;
  else PASS when all three systems pass; FAIL when any fails; else INCOMPLETE.
- Provenance (an M-d2 record that fails any of these is not the protocol's: `run` refuses
  to make it, before any timing, and `judge` refuses the whole records file):
  the calibration file's sha256 is REGISTERED_CALIB; calib_check.ok, i.e. the loaded host
  library's shipped (build-time embedded) calibration equals that file's, as canonical text
  and as bit-identical predictions (in-line DEFAULT/COLD and BURST/REPEAT, and the pack, at
  every level); registered_check, i.e. the loaded host library reproduces the system's
  registered bar (rank's choice and g = N, to 3 decimals) and its p90 class (`predict`'s
  check, so a model-code change under an unchanged calibration is caught); dirty is false
  (no tracked change, this driver included); md2_cleared cites the closure's commit (an
  ancestor of the record's commit) and NOTES-C1 section; driver_sha256 equals the judging
  driver's own sha256 (so the judge is the committed driver that made the record; NOTES-C1
  §4.3 names the registered one); the protocol's --reps 15, --attempts 2, --warmup-s 3,
  --primer-mib 1024, --seed 1 and the 5 % spread rule; every level's configuration is the
  bars': N moduli, g = N, GENERATE, in-line (packed NONE), RN, EMULATE. Per file, the M-d2
  records share one library: one (lib_sha256, host_lib_sha256, host_api). `run` also
  refuses an M-d2 system unless --lib is ROOT/build/libimmerdink.so.
- Reported: per phase of S14 measured vs predicted (DEFAULT/COLD and BURST/REPEAT), TFLOPS-eq
  first with time alongside; the paired per-rep S14/F14 ratios; S9/F9 and t14/t8, t14/t9
  (the gain against fixed N = 8-9); P14/PF14, P14/F14 and each re-pack's time against
  imd_perf_predict_pack; the bit-exact check; queued_ahead (pooled, and the gated S14 / F14
  runs not queued ahead, counted apart); spreads; clocks; each record's admissibility and
  role in the decision.
TFLOPS-eq = 2 m n k (the GEMM view; FP64 work the call replaces) / time.
"""

import argparse
import ctypes as C
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(ROOT, "python"), HERE]

LEVELS = (6, 8, 9, 10, 14)
RUNS = {                         # name -> ([(iterations, N)], packed)
    "S14": ([(5, 6), (4, 8), (4, 10), (3, 14)], False),
    "F14": ([(16, 14)], False),
    "S9": ([(6, 6), (10, 9)], False),
    "F9": ([(16, 9)], False),
    "F8": ([(16, 8)], False),
    "P14": ([(5, 6), (4, 8), (4, 10), (3, 14)], True),
    "PF14": ([(16, 14)], True),
}
ORDER = ("S14", "F14", "S9", "F9", "F8", "P14", "PF14")

# name: spec, extents, invariant operand, GEMM view (m, n, k), B's layout (K modes first:
# "kn", else "nk"), registered bar (None: reported only), the class's p90 (DESIGN §6c.5)
SYSTEMS = {
    "ppl": dict(spec="ijcd,abcd->ijab", ext=dict(i=10, j=10, a=96, b=96, c=96, d=96),
                inv="B", mnk=(100, 9216, 9216), b_layout="nk", bar=0.750, p90=0.15),
    "df": dict(spec="acQ,bdQ->acbd", ext=dict(a=64, b=64, c=64, d=64, Q=2032),
               inv="B", mnk=(4096, 4096, 2032), b_layout="nk", bar=0.664, p90=0.15),
    "sq4096": dict(spec="ik,kj->ij", ext=dict(i=4096, k=4096, j=4096),
                   inv="A", mnk=(4096, 4096, 4096), b_layout="kn", bar=0.673, p90=0.15),
    "jacobi": dict(spec="abcd,cdij->abij", ext=dict(a=40, b=40, c=40, d=40, i=10, j=10),
                   inv="A", mnk=(1600, 100, 1600), b_layout="kn", bar=None, predicted=0.786,
                   p90=0.30),
    # tooling checks only, not M-d2 systems: a square with A invariant, and a small PPL-like
    # tensor contraction with B invariant (GEMM view 100 x 576 x 576)
    "smoke": dict(spec="ik,kj->ij", ext=dict(i=512, k=512, j=512),
                  inv="A", mnk=(512, 512, 512), b_layout="kn", bar=None, p90=None),
    "smoke_t": dict(spec="ijcd,abcd->ijab", ext=dict(i=10, j=10, a=24, b=24, c=24, d=24),
                    inv="B", mnk=(100, 576, 576), b_layout="nk", bar=None, p90=None),
}
MD2_SYSTEMS = ("ppl", "df", "sq4096", "jacobi")
GATED = ("ppl", "df", "sq4096")
DATA = ("normal", "decay_opp")      # both gated (MD2-3)
CAP = 0.8
CLASSES = ((1e-3, ">= 1 ms", 0.15), (1e-4, "100 us - 1 ms", 0.20), (0.0, "< 100 us", 0.30))
# The calibration the bars were derived with (src/host/calib/rtx4090_sm89.calib, shipped
# since v1.18; NOTES-C1 §5.9 MC-1: unchanged after M-c).
REGISTERED_CALIB = "d31f62ca5e8d8893bed427f5a6e1a7905e1acf008ab1f4743cc2dcbb2a60e933"
CALIB_FILE = os.path.join(ROOT, "src", "host", "calib", "rtx4090_sm89.calib")
# The bars' configuration at every level N (with num_moduli = g = N; MD2-7)
BAR_CONFIG = dict(vec_mode="GENERATE", packed="NONE", rounding="RN", mode="EMULATE")
MAX_SPREAD, ATTEMPTS, WARMUP_S, REPS, PHI, SEED, PRIMER_MIB = 5.0, 2, 3.0, 15, 0.5, 1, 1024
MAX_RECORDS = 2                     # per (system, data): the first and one re-run
# The one library the M-d2 systems run on (built at the registered commit; NOTES-C1 §4.3)
DEFAULT_LIB = os.path.join(ROOT, "build", "libimmerdink.so")
# --md2-cleared cites the commit and the NOTES-C1 section that record MC-3's closure
CLEARED_COMMIT = re.compile(r"\b[0-9a-f]{7,40}\b")
CLEARED_SECTION = re.compile(r"NOTES-C1 §\d+(\.\d+)*")


# The bit-exact check fills C with this NaN (a quiet NaN with a payload no kernel produces)
# before each call, so that a call that writes nothing leaves it in place and is detected.
NAN_BITS = 0x7FF8DEADBEEF0001


def bitexact_level(ref_bits, got_bits):
    """(bit-exact, in-line elements still holding NAN_BITS, packed elements still holding it)
    for one level; the arguments are numpy uint64 arrays (C's bits after each call). Bit-exact
    only if the in-line call left no pattern (it wrote every element) and the packed result
    equals the in-line one."""
    ref_left, got_left = int((ref_bits == NAN_BITS).sum()), int((got_bits == NAN_BITS).sum())
    equal = bool((ref_bits == got_bits).all())
    return (ref_left == 0 and equal), ref_left, got_left


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


DRIVER_SHA256 = sha256(os.path.abspath(__file__))


def git(*args):
    r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def git_is_ancestor(cited, commit):
    """True when the commit `cited` names is `commit` or one of its ancestors."""
    if not commit:
        return False
    r = subprocess.run(["git", "-C", ROOT, "merge-base", "--is-ancestor", cited, commit],
                       capture_output=True, text=True)
    return r.returncode == 0


def provenance(lib_path, host_lib_path):
    status = git("status", "--porcelain", "--untracked-files=no")
    out = {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status),
           "host_lib": os.path.realpath(host_lib_path), "host_lib_sha256": sha256(host_lib_path),
           "calib_file_sha256": sha256(CALIB_FILE), "driver_sha256": DRIVER_SHA256}
    if lib_path:
        out |= {"lib": os.path.realpath(lib_path), "lib_sha256": sha256(lib_path)}
    return out


def calib_check(hl, names):
    """The functional calibration check: the loaded host library's shipped calibration (the
    one embedded at build time, which every calib=None query uses) equals the calibration file
    in the tree (whose sha256 is checked apart), as canonical text (imd_perf_calib_format)
    and as bit-identical predictions on `names` at every level (in-line g = N GENERATE,
    DEFAULT/COLD and BURST/REPEAT, and the invariant operand's pack)."""
    from immerdink import host, planner
    reg = host.Calibration.load(CALIB_FILE, hl)
    text_d, text_r = host.Calibration.default(hl).format(), reg.format()
    pl = planner.Planner(lib=hl)
    diff = []
    for name in names:
        s = SYSTEMS[name]
        q = pl.query(s["spec"], s["ext"])
        op = 0 if s["inv"] == "A" else 1
        for N in LEVELS:
            for kw in (dict(), dict(regime="BURST", cache="REPEAT")):
                a = q.predict(N, N, "GENERATE", **kw).time_s
                b = q.predict(N, N, "GENERATE", calib=reg, **kw).time_s
                if a != b:
                    diff.append(f"{name} N = {N} {kw or 'DEFAULT/COLD'}: {a!r} != {b!r}")
            spec = q.pack_desc_for(op, N, N, "GENERATE")
            a, b = q.predict_pack(spec).time_s, q.predict_pack(spec, calib=reg).time_s
            if a != b:
                diff.append(f"{name} N = {N} pack: {a!r} != {b!r}")
    return dict(ok=text_d == text_r and not diff, text_equal=text_d == text_r,
                predictions_equal=not diff, prediction_diffs=diff[:10], systems=list(names),
                shipped_text_sha256=hashlib.sha256(text_d.encode()).hexdigest(),
                file_text_sha256=hashlib.sha256(text_r.encode()).hexdigest())


def level_config(ch):
    """A PrecisionPlan choice's configuration, as recorded and checked."""
    return dict(num_moduli=ch.num_moduli, g=ch.residue_group, vec_mode=ch.vec_mode,
                packed=ch.packed, rounding=ch.rounding, mode=ch.mode)


def config_problems(levels):
    """Why a record's per-level configuration ({str(N): dict}) is not the bars' (MD2-7):
    num_moduli = g = N, GENERATE, in-line (packed NONE), RN, EMULATE. [] if it is."""
    bad = []
    for N in LEVELS:
        v = (levels or {}).get(str(N))
        if v is None:
            bad.append(f"N = {N}: no configuration")
            continue
        for k, want in (dict(num_moduli=N, g=N) | BAR_CONFIG).items():
            if v.get(k) != want:
                bad.append(f"N = {N}: {k} {v.get(k)!r}, not {want!r}")
    return bad


def cleared_problems(text, commit, ancestor=git_is_ancestor):
    """Why an --md2-cleared reason does not cite MC-3's closure ([] if it does): it names a
    NOTES-C1 section ("NOTES-C1 §n.m") and a commit that is `commit` or an ancestor of it
    (the closure was committed before the run)."""
    text = str(text or "").strip()
    if not text:
        return ["no --md2-cleared reason"]
    bad = []
    if not CLEARED_SECTION.search(text):
        bad.append("--md2-cleared cites no NOTES-C1 section ('NOTES-C1 §n.m') recording "
                   "MC-3's closure")
    if not any(ancestor(c, commit) for c in CLEARED_COMMIT.findall(text)):
        bad.append("--md2-cleared cites no commit recording MC-3's closure that is the "
                   f"record's commit ({str(commit)[:7]}) or an ancestor of it")
    return bad


def provenance_problems(rec, ancestor=git_is_ancestor):
    """Why an M-d2 record is not the pre-registered protocol's ([] if it is): the judge
    refuses a records file that holds one, and `run` refuses to make one."""
    bad = []
    if rec.get("calib_file_sha256") != REGISTERED_CALIB:
        bad.append(f"calibration file {rec.get('calib_file_sha256')} is not the registered "
                   "one; re-derive and commit the bars first (IR23)")
    cc = rec.get("calib_check")
    if not cc:
        bad.append("no calib_check")
    elif cc.get("ok") is not True:
        bad.append("the library's shipped calibration is not the registered file's "
                   "(calib_check)")
    rc = rec.get("registered_check")
    if not rc:
        bad.append("no registered_check")
    elif rc.get("same") is not True or rc.get("p90_same") is not True:
        bad.append(f"the host library does not reproduce the registered bar "
                   f"{rc.get('registered')!r} and p90 (registered_check: derived "
                   f"{rc.get('derived')!r}, p90 {rc.get('p90')!r})")
    if rec.get("dirty") is not False:
        bad.append(f"tree dirty = {rec.get('dirty')!r}, not false")
    bad += cleared_problems(rec.get("md2_cleared"), rec.get("commit"), ancestor)
    if rec.get("driver_sha256") != DRIVER_SHA256:
        bad.append(f"driver {str(rec.get('driver_sha256'))[:12]}… is not this driver "
                   f"({DRIVER_SHA256[:12]}…)")
    pr = rec.get("protocol") or {}
    fixed = dict(reps=REPS, attempts=ATTEMPTS, warmup_s=WARMUP_S, primer_mib=PRIMER_MIB,
                 seed=SEED, max_spread_pct=MAX_SPREAD)
    bad += [f"protocol {k} = {pr.get(k)!r}, not {v!r}" for k, v in fixed.items()
            if pr.get(k) != v]
    bad += config_problems(rec.get("levels"))
    return bad


def library_problems(recs):
    """Why the M-d2 records of one file do not share one library ([] if they do): one
    (lib_sha256, host_lib_sha256, host_api), each recorded."""
    ids = sorted({(r.get("lib_sha256"), r.get("host_lib_sha256"), r.get("host_api"))
                  for r in recs if r["system"] in MD2_SYSTEMS}, key=repr)
    bad = []
    if any(None in t for t in ids):
        bad.append("an M-d2 record without lib_sha256, host_lib_sha256 or host_api")
    if len(ids) > 1:
        bad.append(f"the M-d2 records come from {len(ids)} libraries: " + "; ".join(
            f"lib {str(a)[:12]}…, host lib {str(b)[:12]}…, host API {c}" for a, b, c in ids))
    return bad


def file_problems(recs, ancestor=git_is_ancestor):
    """Why `judge` refuses a records file: [(where, problems)], [] if it judges it."""
    out = [(f"record {i + 1} ({r['system']} {r['data']})", p) for i, r in enumerate(recs)
           if r["system"] in MD2_SYSTEMS for p in (provenance_problems(r, ancestor),) if p]
    lp = library_problems(recs)
    if lp:
        out.append(("the M-d2 records", lp))
    return out


def inadmissible(rec):
    """Why a record is inadmissible, neither PASS nor FAIL (rule (a)); [] if admissible:
    contaminated not false, or the bit-exact check not true at every level."""
    why = []
    if rec.get("contaminated") is not False:
        why.append(f"contaminated (worst spread > {MAX_SPREAD:g} %)")
    be = rec.get("bitexact_packed_vs_inline") or {}
    if bitexact_failed(rec):
        why.append("packed != in-line at N = " + ", ".join(str(N) for N in bitexact_failed(rec)))
    missing = [N for N in LEVELS if be.get(str(N)) is not True and be.get(str(N)) is not False]
    if missing:
        why.append("no bit-exact result (true / false) at N = " + ", ".join(map(str, missing)))
    return why


def bitexact_failed(rec):
    """The levels whose packed-vs-in-line check failed: a library defect (stops M-d2)."""
    be = rec.get("bitexact_packed_vs_inline") or {}
    return [N for N in LEVELS if be.get(str(N)) is False]


def stopped(recs):
    """The records whose bit-exact failure stops M-d2: any record in the file, an M-d2
    system's or a smoke's (rule (a))."""
    return [r for r in recs if bitexact_failed(r)]


def s14_f14(rec):
    """The gated statistic: median of the S14 run totals / median of the F14 run totals."""
    reps = rec["reps"]
    return (statistics.median(x["S14"]["total_ms"] for x in reps) /
            statistics.median(x["F14"]["total_ms"] for x in reps))


def decide(recs):
    """Rules (a)-(d) over the records in file order. Returns (roles, pairs, systems, md2):
    roles[i] = (role, reasons) for every record (role: decisive / inadmissible / after the
    decisive record / excess); pairs[(system, data)] = INSIDE / OUTSIDE / INCOMPLETE (with
    why) for the M-d2 systems; systems[name] = PASS / FAIL / INCOMPLETE for the gated three;
    md2 = PASS / FAIL / INCOMPLETE / STOPPED."""
    roles, idx = [None] * len(recs), {}
    for i, r in enumerate(recs):
        idx.setdefault((r["system"], r["data"]), []).append(i)
    stop = bool(stopped(recs))
    pairs = {}
    for key, ii in idx.items():
        decisive = None
        for j, i in enumerate(ii):
            why = inadmissible(recs[i])
            if j >= MAX_RECORDS:
                roles[i] = ("excess: past the one re-run, not considered", why)
            elif decisive is not None:
                roles[i] = ("after the decisive record, not considered", why)
            elif why:
                roles[i] = ("inadmissible", why)
            else:
                decisive = i
                roles[i] = ("decisive", why)
        if key[0] not in MD2_SYSTEMS:
            continue
        b = band(key[0])
        if decisive is not None:
            r = s14_f14(recs[decisive])
            pairs[key] = dict(outcome="INSIDE" if b["lo"] <= r <= b["hi"] else "OUTSIDE",
                              record=decisive, S14_F14=r)
        else:
            used = min(len(ii), MAX_RECORDS)
            pairs[key] = dict(outcome="INCOMPLETE", record=None, S14_F14=None,
                              why="both attempts inadmissible: record the cause; a further "
                                  "attempt needs a recorded reviewer verdict or Eric's "
                                  "decision (NOTES-C1 §4.3)" if used == MAX_RECORDS else
                                  "M-d2 STOPPED by a bit-exact failure: no re-run; a restart "
                                  "after the fix needs a recorded reviewer verdict or Eric's "
                                  "decision (NOTES-C1 §4.3)" if stop else
                                  "no admissible record yet (one re-run allowed)")
    systems = {}
    for name in GATED:
        o = [pairs.get((name, d), {}).get("outcome", "INCOMPLETE") for d in DATA]
        systems[name] = ("PASS" if all(x == "INSIDE" for x in o) else
                         "FAIL" if "OUTSIDE" in o else "INCOMPLETE")
    if stop:
        md2 = "STOPPED"
    else:
        md2 = ("PASS" if all(x == "PASS" for x in systems.values()) else
               "FAIL" if "FAIL" in systems.values() else "INCOMPLETE")
    return roles, pairs, systems, md2


def rerun_problems(prior, name, data):
    """Why `run` may not add a record of (name, data) to a records file holding `prior`
    (rules (a)-(c)); [] if it may."""
    bad = []
    stop = stopped(prior)
    if stop:
        bad.append(f"M-d2 is STOPPED: the records hold a bit-exact failure ({stop[0]['system']} "
                   f"{stop[0]['data']}), a library defect (rule (a)); a restart after the fix "
                   "needs a recorded reviewer verdict or Eric's decision (NOTES-C1 §4.3)")
    same = [r for r in prior if (r["system"], r["data"]) == (name, data)]
    if any(not inadmissible(r) for r in same):
        bad.append(f"{name} {data} is decided: the records hold an admissible one (rule (c))")
    elif len(same) >= MAX_RECORDS:
        bad.append(f"{name} {data}: the one re-run is used (rule (b)); record the cause of the "
                   "inadmissible attempts; a further attempt needs a recorded reviewer verdict "
                   "or Eric's decision (NOTES-C1 §4.3)")
    return bad


def read_records(path):
    recs = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("kind") == "md2_run":
                        recs.append(d)
    return recs


def cls(t):
    return next((name, p) for lo, name, p in CLASSES if t >= lo)


def flops(name):
    m, n, k = SYSTEMS[name]["mnk"]
    return 2.0 * m * n * k


# ------------------------------------------------------------------------------------------
# [M] predictions (CPU)

def predictions(name, host_lib):
    """Per level: PrecisionPlan's configuration and the [M] times; the runs' predicted totals;
    the predicted ratios. DEFAULT/COLD is the bars' regime; BURST/REPEAT §6c.5's."""
    from immerdink import planner, precision
    s = SYSTEMS[name]
    pl = planner.Planner(lib=host_lib)
    q = pl.query(s["spec"], s["ext"])
    cf = precision.PrecisionPlan(pl, {name: (s["spec"], s["ext"])}, levels=LEVELS,
                                 vec_modes=("GENERATE",), monotone=False).configs()[name]
    op = 0 if s["inv"] == "A" else 1
    lv = {}
    for N in LEVELS:
        ch = cf[N]
        g, vec = ch.residue_group, ch.vec_mode
        spec = q.pack_desc_for(op, N, g, vec)
        packing = (spec, None) if op == 0 else (None, spec)
        pk = q.predict_pack(spec)
        lv[N] = dict(N=N, g=g, vec_mode=vec, status=ch.status,
                     t_cold=ch.time_s,                                   # = rank's, in-line
                     t_gN_cold=q.predict(N).time_s,                      # M-b's g = N view
                     t_burst_repeat=q.predict(N, g, vec, regime="BURST", cache="REPEAT").time_s,
                     t_packed_cold=q.predict(N, g, vec, packing=packing).time_s,
                     pack_cold=pk.time_s, pack_bytes=pk.pack_bytes,
                     workspace_bytes=ch.workspace_bytes)
        lv[N]["class"] = cls(lv[N]["t_cold"])[0]

    def tot(run, key):
        steps, packed = RUNS[run]
        t = sum(n * lv[N][key if not packed else "t_packed_cold"] for n, N in steps)
        if packed:
            t += sum(lv[N]["pack_cold"] for _, N in steps)
        return t
    totals = {k: {r: tot(r, k) for r in RUNS} for k in ("t_cold", "t_gN_cold", "t_burst_repeat")}
    rat = {k: dict(S14_F14=v["S14"] / v["F14"], S9_F9=v["S9"] / v["F9"],
                   P14_PF14=v["P14"] / v["PF14"], P14_F14=v["P14"] / v["F14"])
           for k, v in totals.items()}
    return dict(system=name, spec=s["spec"], ext=s["ext"], mnk=s["mnk"], invariant=s["inv"],
                flops_eq=flops(name), levels={str(N): v for N, v in lv.items()},
                totals=totals, ratios=rat)


def registered_check(name, p):
    """`predict`'s check of an M-d2 system on the loaded host library (p = predictions()):
    the registered bar (the Jacobi view: its registered prediction) reproduces to 3 decimals
    from rank's choice and from g = N, and the levels' classes give the registered p90."""
    s = SYSTEMS[name]
    reg = s["bar"] if s["bar"] is not None else s["predicted"]
    r = p["ratios"]
    got = {k: round(r[k]["S14_F14"], 3) for k in ("t_cold", "t_gN_cold")}
    same = all(abs(v - reg) < 5e-4 for v in got.values())
    classes = {p["levels"][str(N)]["class"] for N in LEVELS}
    p90 = max(cls(p["levels"][str(N)]["t_cold"])[1] for N in LEVELS)
    return dict(registered=reg, derived=got, same=same, classes=sorted(classes), p90=p90,
                p90_same=p90 == s["p90"])


def band(name):
    s = SYSTEMS[name]
    r = s["bar"] if s["bar"] is not None else s.get("predicted")
    if r is None:
        return None
    lo, hi = r * (1 - s["p90"]), r * (1 + s["p90"])
    if name in GATED:
        hi = min(hi, CAP)
    return dict(ratio=r, p90=s["p90"], lo=lo, hi=hi, cap=CAP if name in GATED else None,
                gated=name in GATED)


def cmd_predict(a):
    from immerdink import host
    hl = host.load(a.host_lib)
    out = dict(provenance=provenance(None, a.host_lib), systems={})
    ok = True
    print("M-d2 predictions [M] (shipped calibration; levels 6, 8, 9, 10, 14; in-line; "
          "GENERATE). TFLOPS-eq first, time alongside.")
    for name in MD2_SYSTEMS + ("smoke", "smoke_t"):
        p = predictions(name, hl)
        b = band(name)
        p["band"] = b
        out["systems"][name] = p
        print(f"\n{name}: {p['spec']} {p['ext']} (GEMM view {p['mnk']}), invariant "
              f"{p['invariant']}")
        print(f"  {'N':>3} {'g':>3} {'TF COLD':>8} {'ms':>8} | {'TF B/R':>7} {'ms':>8} | "
              f"{'packed ms':>9} {'pack ms':>8} {'class':>14}")
        for N in LEVELS:
            v = p["levels"][str(N)]
            fe = p["flops_eq"]
            print(f"  {N:>3} {v['g']:>3} {fe / v['t_cold'] * 1e-12:>8.2f} "
                  f"{1e3 * v['t_cold']:>8.3f} | {fe / v['t_burst_repeat'] * 1e-12:>7.2f} "
                  f"{1e3 * v['t_burst_repeat']:>8.3f} | {1e3 * v['t_packed_cold']:>9.3f} "
                  f"{1e3 * v['pack_cold']:>8.3f} {v['class']:>14}")
        r = p["ratios"]
        print(f"  S14/F14: {r['t_cold']['S14_F14']:.4f} (rank's choice, DEFAULT/COLD), "
              f"{r['t_gN_cold']['S14_F14']:.4f} (g = N), {r['t_burst_repeat']['S14_F14']:.4f} "
              f"(BURST/REPEAT); S9/F9 {r['t_cold']['S9_F9']:.4f}; P14/PF14 "
              f"{r['t_cold']['P14_PF14']:.4f}; P14/F14 {r['t_cold']['P14_F14']:.4f}")
        if b:
            print(f"  band: {b['ratio']:.3f} x (1 -/+ {b['p90']:.2f}) = [{b['lo']:.4f}, "
                  f"{b['hi']:.4f}]{' (capped at 0.8)' if b['cap'] else ''}"
                  f"{'' if b['gated'] else ', reported only'}")
        if name in MD2_SYSTEMS:
            rc = p["registered_check"] = registered_check(name, p)
            print(f"  registered {rc['registered']:.3f}: derived {rc['derived']} -> "
                  f"{'same' if rc['same'] else 'DIFFERENT'}; classes {rc['classes']} -> p90 "
                  f"{rc['p90']:.2f} {'as registered' if rc['p90_same'] else 'DIFFERENT'}")
            ok &= rc["same"] and rc["p90_same"]
    calib_ok = out["provenance"]["calib_file_sha256"] == REGISTERED_CALIB
    cc = out["calib_check"] = calib_check(hl, MD2_SYSTEMS + ("smoke", "smoke_t"))
    out["registered_calib"] = REGISTERED_CALIB
    out["calib_is_registered"] = calib_ok
    out["bars_reproduced"] = bool(ok)
    print(f"\ncalibration file {'is' if calib_ok else 'is NOT'} the registered one; the "
          f"library's shipped calibration {'equals' if cc['ok'] else 'DIFFERS FROM'} it "
          f"(canonical text {cc['text_equal']}, predictions {cc['predictions_equal']}); bars " +
          ("reproduced" if ok else
           "NOT reproduced: re-derive and commit them before M-d2 (IR23)"))
    print(f"driver sha256 {DRIVER_SHA256}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0 if ok and calib_ok and cc["ok"] else 1


# ------------------------------------------------------------------------------------------
# The GPU run

class Rt:
    """The CUDA runtime calls this benchmark needs (libcudart of libimmerdink)."""

    def __init__(self):
        rt = self.rt = C.CDLL("libcudart.so.13")
        for fn, args in {
            "cudaMalloc": [C.POINTER(C.c_void_p), C.c_size_t], "cudaFree": [C.c_void_p],
            "cudaMemcpy": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int],
            "cudaMemsetAsync": [C.c_void_p, C.c_int, C.c_size_t, C.c_void_p],
            "cudaDeviceSynchronize": [], "cudaStreamCreate": [C.POINTER(C.c_void_p)],
            "cudaStreamSynchronize": [C.c_void_p],
            "cudaEventCreate": [C.POINTER(C.c_void_p)],
            "cudaEventRecord": [C.c_void_p, C.c_void_p],
            "cudaEventSynchronize": [C.c_void_p],
            "cudaEventElapsedTime": [C.POINTER(C.c_float), C.c_void_p, C.c_void_p],
            "cudaGetErrorString": [C.c_int],
        }.items():
            getattr(rt, fn).argtypes = args
            getattr(rt, fn).restype = C.c_int
        rt.cudaGetErrorString.restype = C.c_char_p

    def ok(self, err, what):
        if err:
            raise RuntimeError(f"{what}: {self.rt.cudaGetErrorString(err).decode()}")

    def malloc(self, n):
        p = C.c_void_p()
        self.ok(self.rt.cudaMalloc(C.byref(p), max(int(n), 1)), "cudaMalloc")
        return p.value

    def upload(self, p, arr):
        self.ok(self.rt.cudaMemcpy(p, arr.ctypes.data, arr.nbytes, 1), "cudaMemcpy H2D")

    def download(self, arr, p):
        self.ok(self.rt.cudaMemcpy(arr.ctypes.data, p, arr.nbytes, 2), "cudaMemcpy D2H")

    def stream(self):
        s = C.c_void_p()
        self.ok(self.rt.cudaStreamCreate(C.byref(s)), "cudaStreamCreate")
        return s.value

    def event(self):
        e = C.c_void_p()
        self.ok(self.rt.cudaEventCreate(C.byref(e)), "cudaEventCreate")
        return e.value

    def record(self, e, s):
        self.ok(self.rt.cudaEventRecord(e, s), "cudaEventRecord")

    def ms(self, a, b):
        self.ok(self.rt.cudaEventSynchronize(b), "cudaEventSynchronize")
        x = C.c_float()
        self.ok(self.rt.cudaEventElapsedTime(C.byref(x), a, b), "cudaEventElapsedTime")
        return x.value

    def sync(self, s):
        self.ok(self.rt.cudaStreamSynchronize(s), "cudaStreamSynchronize")


def clock_state():
    q = ("clocks.sm,clocks.max.sm,power.draw,temperature.gpu,pstate,clocks_event_reasons.active,"
         "clocks_event_reasons.applications_clocks_setting,clocks_event_reasons.sw_power_cap,"
         "clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_slowdown")
    try:
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        vals = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return dict(zip(q.split(","), vals))
    except Exception as e:  # noqa: BLE001  (a record field, not a failure)
        return {"error": repr(e)}


def operands(name, data, seed):
    """(A, B) as dense host arrays of the spec's operand shapes."""
    import numpy as np
    s = SYSTEMS[name]
    lhs = s["spec"].split("->")[0].split(",")
    shapes = [tuple(s["ext"][c] for c in t) for t in lhs]
    m, n, k = s["mnk"]
    if data == "normal":
        out = []
        for x, sh in enumerate(shapes):
            r = np.random.default_rng([seed, x])
            size = math.prod(sh)
            out.append(((r.random(size) - 0.5) * np.exp(PHI * r.standard_normal(size)))
                       .reshape(sh))
        return out
    import c2_fixtures
    A, Bt = c2_fixtures.synthetic("decay_opp", m, n, k, seed)     # (m, k), (n, k)
    B = Bt if s["b_layout"] == "nk" else np.ascontiguousarray(Bt.T)
    return [np.ascontiguousarray(A).reshape(shapes[0]), B.reshape(shapes[1])]


def spread(xs):
    return 100.0 * (max(xs) - min(xs)) / statistics.median(xs)


def cmd_run(a):
    import numpy as np
    import immerdink as imd
    from immerdink import planner, precision
    name = a.system
    if name in MD2_SYSTEMS and not a.md2_cleared:
        print(f"{name} is an M-d2 system: M-d2 waits until M-c's MC-3 path closes (NOTES-C1 "
              "§5.9 'M-d2 ordering', binding). Once it has, pass --md2-cleared citing the "
              "commit and the NOTES-C1 section that record the closure.", file=sys.stderr)
        return 2
    if name in MD2_SYSTEMS and os.path.realpath(a.lib) != os.path.realpath(DEFAULT_LIB):
        print(f"{name} is an M-d2 system: it runs on {DEFAULT_LIB} only (built at the "
              f"registered commit; NOTES-C1 §4.3), not {a.lib}", file=sys.stderr)
        return 2
    s = SYSTEMS[name]
    lib = imd.load_library(a.lib)
    hl = lib.host
    rec = dict(kind="md2_run", system=name, data=a.data, spec=s["spec"], ext=s["ext"],
               mnk=s["mnk"], invariant=s["inv"], flops_eq=flops(name), md2_cleared=a.md2_cleared,
               date=time.strftime("%Y-%m-%dT%H:%M:%S"), host_api=lib.version,
               protocol=dict(levels=LEVELS, runs={k: v[0] for k, v in RUNS.items()},
                             reps=a.reps, primer_mib=a.primer_mib, warmup_s=a.warmup_s,
                             max_spread_pct=MAX_SPREAD, attempts=a.attempts, seed=a.seed,
                             phi=PHI if a.data == "normal" else None))
    rec |= provenance(lib.path, os.path.join(os.path.dirname(lib.path), "libimmerdink_host.so"))
    rec["calib_check"] = calib_check(hl, (name,))
    pl = planner.Planner(lib=hl)
    cf = precision.PrecisionPlan(pl, {name: (s["spec"], s["ext"])}, levels=LEVELS,
                                 vec_modes=("GENERATE",), monotone=False).configs()[name]
    rec["levels"] = {str(N): level_config(cf[N]) for N in LEVELS}
    if name in MD2_SYSTEMS:
        rec["registered_check"] = registered_check(name, predictions(name, hl))
        # Provenance and rules (b) / (a) before any GPU work: a record that would not be
        # admissible protocol evidence is not made (judge refuses the file otherwise).
        prior = read_records(a.out)
        problems = provenance_problems(rec)
        problems += library_problems(prior + [rec])
        problems += rerun_problems(prior, name, a.data)
        if problems:
            print(f"{name} {a.data}: not run:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 2
    rt = Rt()
    st = rt.stream()
    A, B = operands(name, a.data, a.seed)
    lhs, rhs = s["spec"].split("->")
    sa, sb = (tuple(s["ext"][c] for c in t) for t in lhs.split(","))
    cshape = tuple(s["ext"][c] for c in rhs)
    assert A.shape == sa and B.shape == sb, (A.shape, sa, B.shape, sb)
    dA, dB, dC = rt.malloc(A.nbytes), rt.malloc(B.nbytes), rt.malloc(8 * math.prod(cshape))
    rt.upload(dA, np.ascontiguousarray(A))
    rt.upload(dB, np.ascontiguousarray(B))
    del A, B
    xA, xB, xC = imd.DeviceArray(dA, sa), imd.DeviceArray(dB, sb), imd.DeviceArray(dC, cshape)
    op = 0 if s["inv"] == "A" else 1
    X, dX = (xA, dA) if op == 0 else (xB, dB)
    plans, pplans, packs = {}, {}, {}
    for N in LEVELS:
        opts = cf[N].options()
        plans[N] = imd.Plan(s["spec"], xA, xB, xC, lib=lib, **opts)
        packs[N] = imd.Pack(plans[N].pack_spec(op), lib=lib)
        pk = (packs[N], None) if op == 0 else (None, packs[N])
        pplans[N] = imd.Plan(s["spec"], xA, xB, xC, lib=lib, packs=pk, **opts)
    nws = max(max(p.workspace_bytes for p in plans.values()),
              max(p.workspace_bytes for p in pplans.values()))
    npk = max(p.pack_bytes for p in packs.values())
    npw = max(p.workspace_bytes for p in packs.values())
    ws, pbuf, pws = rt.malloc(nws), rt.malloc(npk), rt.malloc(npw)
    primer_bytes = a.primer_mib << 20
    scratch = rt.malloc(primer_bytes)
    rec["levels"] = {str(N): dict(N=N, **level_config(cf[N]),
                                  workspace_bytes=plans[N].workspace_bytes,
                                  packed_workspace_bytes=pplans[N].workspace_bytes,
                                  pack_bytes=packs[N].pack_bytes,
                                  pack_workspace_bytes=packs[N].workspace_bytes)
                     for N in LEVELS}
    rec["buffers"] = dict(workspace=nws, packed=npk, pack_workspace=npw, primer=primer_bytes)
    wsa = imd.DeviceArray(ws, (nws,), dtype="u1")
    pwa = imd.DeviceArray(pws, (npw,), dtype="u1")

    # the bit-exact check: the checked calls, in-line against packed (PackedBuffer), per level
    pb = imd.PackedBuffer(pbuf, npk)
    nC = math.prod(cshape)
    ref, got = np.empty(nC), np.empty(nC)
    nan_c = np.full(nC, NAN_BITS, dtype=np.uint64).view(np.float64)
    rec["bitexact_packed_vs_inline"] = {}
    rec["bitexact_unwritten"] = {}  # per level: elements still holding NAN_BITS, in-line / packed
    for N in LEVELS:
        rt.upload(dC, nan_c)  # C := NaN pattern, then the in-line reference call
        plans[N].contract(xA, xB, xC, workspace=wsa, stream=st)
        rt.sync(st)
        rt.download(ref, dC)
        packs[N].pack(X, pb, workspace=pwa, stream=st)
        rt.sync(st)
        rt.upload(dC, nan_c)  # C := NaN pattern again, then the packed call
        pplans[N].contract(xA, xB, xC, workspace=wsa, stream=st,
                           packed=(pb, None) if op == 0 else (None, pb))
        rt.sync(st)
        rt.download(got, dC)
        ok, ref_left, got_left = bitexact_level(ref.view(np.uint64), got.view(np.uint64))
        rec["bitexact_packed_vs_inline"][str(N)] = ok
        rec["bitexact_unwritten"][str(N)] = [ref_left, got_left]

    def contract(N, packed):
        if packed:
            pr = (pbuf, packs[N].pack_bytes)
            pplans[N].contract_ptr(1.0, dA, dB, 0.0, dC, ws, nws, st,
                                   packed=(pr, None) if op == 0 else (None, pr))
        else:
            plans[N].contract_ptr(1.0, dA, dB, 0.0, dC, ws, nws, st)

    def pack(N):
        packs[N].pack_ptr(dX, pbuf, packs[N].pack_bytes, pws, npw, st)

    # warm-up: the system's own in-line N = 14 call for >= warmup_s, then every level once
    t0, nw = time.perf_counter(), 0
    while True:
        for _ in range(8):
            contract(14, False)
        nw += 8
        rt.sync(st)
        if time.perf_counter() - t0 >= a.warmup_s:
            break
    for N in LEVELS:
        contract(N, False)
        pack(N)
        contract(N, True)
    rt.sync(st)
    rec["warmup"] = dict(seconds=time.perf_counter() - t0, calls=nw)
    rec["clock_after_warmup"] = clock_state()

    ev = [rt.event() for _ in range(2 + max(sum(c for c, _ in v[0]) + len(v[0])
                                            for v in RUNS.values()))]

    def one(run):
        steps, packed = RUNS[run]
        rt.record(ev[0], st)
        rt.ok(rt.rt.cudaMemsetAsync(scratch, 0, primer_bytes, st), "cudaMemsetAsync")
        rt.record(ev[1], st)
        h0 = time.perf_counter()
        i, marks = 2, []
        for count, N in steps:
            if packed:
                pack(N)
                rt.record(ev[i], st)
                marks.append(("pack", N))
                i += 1
            for _ in range(count):
                contract(N, packed)
                rt.record(ev[i], st)
                marks.append(("call", N))
                i += 1
        enq = time.perf_counter() - h0
        rt.sync(st)
        primer = rt.ms(ev[0], ev[1])
        steps_ms = [rt.ms(ev[j - 1], ev[j]) for j in range(2, i)]
        calls = [(N, t) for (kind, N), t in zip(marks, steps_ms) if kind == "call"]
        packs_ms = [(N, t) for (kind, N), t in zip(marks, steps_ms) if kind == "pack"]
        return dict(total_ms=rt.ms(ev[1], ev[i - 1]), primer_ms=primer, enqueue_ms=1e3 * enq,
                    queued_ahead=1e3 * enq < primer, calls=calls, packs=packs_ms)

    attempts = []
    for at in range(a.attempts):
        reps = []
        for r in range(a.reps):
            order = ORDER[r % len(ORDER):] + ORDER[:r % len(ORDER)]
            reps.append({run: one(run) for run in order} | {"order": list(order)})
        worst = max(spread([x[run]["total_ms"] for x in reps]) for run in ("S14", "F14"))
        attempts.append(dict(reps=reps, worst_spread_pct=worst))
        if worst <= MAX_SPREAD:
            break
    best = min(attempts, key=lambda x: x["worst_spread_pct"])
    rec["attempts"] = [dict(worst_spread_pct=x["worst_spread_pct"]) for x in attempts]
    rec["kept_attempt"] = attempts.index(best)
    rec["contaminated"] = best["worst_spread_pct"] > MAX_SPREAD
    rec["reps"] = best["reps"]
    rec["clock_after_reps"] = clock_state()
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    med = {run: statistics.median(x[run]["total_ms"] for x in rec["reps"]) for run in RUNS}
    fe = rec["flops_eq"]
    print(f"{name} {a.data}: " + ", ".join(
        f"{run} {16 * fe / med[run] * 1e-9:.1f} TF-eq ({med[run]:.3f} ms)" for run in ORDER) +
        f"; S14/F14 {med['S14'] / med['F14']:.4f}; bit-exact "
        f"{all(rec['bitexact_packed_vs_inline'].values())}; worst spread "
        f"{best['worst_spread_pct']:.2f} %; " +
        ("admissible" if not inadmissible(rec) else
         "INADMISSIBLE: " + "; ".join(inadmissible(rec))))
    if bitexact_failed(rec):
        print(f"STOP: packed != in-line at N = {bitexact_failed(rec)}, a library defect "
              "(rule (a)); M-d2 is stopped until it is fixed and recorded, and a restart needs "
              "a recorded reviewer verdict or Eric's decision (NOTES-C1 §4.3)", file=sys.stderr)
        return 3
    return 0


# ------------------------------------------------------------------------------------------
# The judge (CPU)

def cmd_judge(a):
    from immerdink import host
    recs = read_records(a.records)
    refused = file_problems(recs)
    if refused:
        for where, p in refused:
            print(f"{where}: not the pre-registered protocol's:\n  " + "\n  ".join(p),
                  file=sys.stderr)
        print(f"{a.records} is not an M-d2 records file: nothing judged. This is a "
              "procedural breach, not a reset: record it in NOTES-C1 "
              "(which record, how it got there); continuing in a new file needs a recorded "
              "reviewer verdict or Eric's decision (NOTES-C1 §4.3)", file=sys.stderr)
        return 2
    roles, pairs, systems, md2 = decide(recs)
    hl = host.load(a.host_lib)
    preds, rows = {}, []
    for i, r in enumerate(recs):
        name = r["system"]
        if name not in preds:
            preds[name] = predictions(name, hl)
        p = preds[name]
        reps = r["reps"]
        med = {run: statistics.median(x[run]["total_ms"] for x in reps) for run in RUNS}
        ratio = med["S14"] / med["F14"]
        paired = [x["S14"]["total_ms"] / x["F14"]["total_ms"] for x in reps]
        b = band(name)
        inside = None if b is None else b["lo"] <= ratio <= b["hi"]
        per_level = {}
        for N in LEVELS:      # every in-line call at N over the reps and runs
            ts = [t for x in reps for run in ("S14", "F14", "S9", "F9", "F8")
                  for n_, t in x[run]["calls"] if n_ == N]
            per_level[N] = statistics.median(ts)
        phases = []
        for count, N in RUNS["S14"][0]:
            ph = statistics.median(sum(t for n_, t in x["S14"]["calls"] if n_ == N) for x in reps)
            lv = p["levels"][str(N)]
            phases.append(dict(N=N, calls=count, ms=ph, tflops_eq=count * r["flops_eq"] / ph * 1e-9,
                               pred_cold_ms=1e3 * count * lv["t_cold"],
                               pred_burst_repeat_ms=1e3 * count * lv["t_burst_repeat"]))
        repack = []
        for N in (6, 8, 10, 14):
            ts = [t for x in reps for n_, t in x["P14"]["packs"] if n_ == N]
            repack.append(dict(N=N, ms=statistics.median(ts),
                               pred_ms=1e3 * p["levels"][str(N)]["pack_cold"]))
        row = dict(record=i + 1, system=name, data=r["data"], role=roles[i][0],
                   reasons=roles[i][1], commit=r.get("commit"), lib_sha256=r.get("lib_sha256"),
                   driver_sha256=r.get("driver_sha256"), host_api=r.get("host_api"),
                   median_ms=med,
                   tflops_eq={run: 16 * r["flops_eq"] / med[run] * 1e-9 for run in RUNS},
                   S14_F14=ratio, paired_median=statistics.median(paired),
                   paired_min=min(paired), paired_max=max(paired),
                   pred_S14_F14=dict(cold=p["ratios"]["t_cold"]["S14_F14"],
                                     burst_repeat=p["ratios"]["t_burst_repeat"]["S14_F14"]),
                   band=b, inside=inside,
                   S9_F9=med["S9"] / med["F9"], pred_S9_F9=p["ratios"]["t_cold"]["S9_F9"],
                   t14_t8=per_level[14] / per_level[8], t14_t9=per_level[14] / per_level[9],
                   P14_PF14=med["P14"] / med["PF14"], P14_F14=med["P14"] / med["F14"],
                   pred_P14_PF14=p["ratios"]["t_cold"]["P14_PF14"],
                   per_level_ms=per_level, phases=phases, repack=repack,
                   bitexact=r.get("bitexact_packed_vs_inline"),
                   queued_ahead=sum(x[run]["queued_ahead"] for x in reps for run in RUNS) /
                   (len(reps) * len(RUNS)),
                   gated_not_queued=sum(not x[run]["queued_ahead"] for x in reps
                                        for run in ("S14", "F14")),
                   gated_runs=2 * len(reps),
                   worst_spread_pct=r["attempts"][r["kept_attempt"]]["worst_spread_pct"],
                   contaminated=r.get("contaminated"), clock=r.get("clock_after_reps"))
        rows.append(row)
    for row in rows:
        print(f"\n## record {row['record']}: {row['system']} ({row['data']}; commit "
              f"{str(row['commit'])[:7]}, lib {str(row['lib_sha256'])[:8]}…, driver "
              f"{str(row['driver_sha256'])[:8]}…, host API {row['host_api']})")
        print(f"role: {row['role']}" + (f" ({'; '.join(row['reasons'])})" if row["reasons"]
                                        else ""))
        print("run   | TFLOPS-eq | ms (median of the reps)")
        for run in ORDER:
            print(f"{run:5s} | {row['tflops_eq'][run]:9.2f} | {row['median_ms'][run]:.3f}")
        b = row["band"]
        print(f"S14/F14 {row['S14_F14']:.4f} (paired median {row['paired_median']:.4f}, "
              f"{row['paired_min']:.4f}-{row['paired_max']:.4f}); predicted "
              f"{row['pred_S14_F14']['cold']:.4f} (DEFAULT/COLD), "
              f"{row['pred_S14_F14']['burst_repeat']:.4f} (BURST/REPEAT)" +
              ("" if b is None else f"; band [{b['lo']:.4f}, {b['hi']:.4f}] -> "
               f"{'INSIDE' if row['inside'] else 'OUTSIDE'}"
               f"{'' if b['gated'] else ' (reported)'}"))
        print("phase of S14 | TFLOPS-eq | ms | predicted ms DEFAULT/COLD (err) | "
              "BURST/REPEAT (err)")
        for ph in row["phases"]:
            print(f"{ph['calls']} x N{ph['N']:<2}      | {ph['tflops_eq']:9.2f} | "
                  f"{ph['ms']:.3f} | {ph['pred_cold_ms']:.3f} "
                  f"({100 * (ph['pred_cold_ms'] / ph['ms'] - 1):+.1f} %) | "
                  f"{ph['pred_burst_repeat_ms']:.3f} "
                  f"({100 * (ph['pred_burst_repeat_ms'] / ph['ms'] - 1):+.1f} %)")
        print(f"S9/F9 {row['S9_F9']:.4f} (pred {row['pred_S9_F9']:.4f}); t14/t8 "
              f"{row['t14_t8']:.3f}, t14/t9 {row['t14_t9']:.3f}; P14/PF14 {row['P14_PF14']:.4f} "
              f"(pred {row['pred_P14_PF14']:.4f}), P14/F14 {row['P14_F14']:.4f}")
        print("re-packs (ms, predicted): " + ", ".join(
            f"N{x['N']} {x['ms']:.3f} ({x['pred_ms']:.3f})" for x in row["repack"]))
        be = row["bitexact"] or {}
        print(f"bit-exact packed = in-line: "
              f"{all(be.get(str(N)) is True for N in LEVELS)}; queued ahead "
              f"{100 * row['queued_ahead']:.0f} % of runs, gated S14/F14 runs not queued ahead "
              f"{row['gated_not_queued']} of {row['gated_runs']}; worst spread "
              f"{row['worst_spread_pct']:.2f} %{' CONTAMINATED' if row['contaminated'] else ''}")
    print("\ndecision (rules (a)-(d), the first admissible record per system and data set):")
    for name in MD2_SYSTEMS:
        for d in DATA:
            v = pairs.get((name, d))
            what = ("no record" if v is None else
                    f"{v['outcome']} (record {v['record'] + 1}, S14/F14 {v['S14_F14']:.4f})"
                    if v["record"] is not None else f"{v['outcome']} ({v['why']})")
            print(f"  {name:6s} {d:9s}: {what}{'' if name in GATED else ' [reported only]'}")
    print(f"\nper system: {systems}; M-d2: {md2}"
          f"{' (a bit-exact failure: a library defect)' if md2 == 'STOPPED' else ''}")
    print(f"judging driver sha256 {DRIVER_SHA256}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(rows=rows, pairs={f"{k[0]}/{k[1]}": v for k, v in pairs.items()},
                           systems=systems, md2=md2, judge_driver_sha256=DRIVER_SHA256,
                           provenance=provenance(None, a.host_lib)), f, indent=1)
    return 0


# ------------------------------------------------------------------------------------------
# The self-test (CPU, standard library): the decision and provenance rules

def cmd_selftest(a):
    """Rules (a)-(d), the bands and the provenance checks on synthetic records."""
    fails, n = [], [0]

    def check(what, got, want):
        n[0] += 1
        if got != want:
            fails.append(f"{what}: got {got!r}, want {want!r}")

    # a stand-in for git: the record's commit, and the one ancestor that records the closure
    head, closure = "1" * 40, "c105ed0" + "0" * 33

    def anc(cited, commit):
        return commit == head and closure.startswith(cited)

    cleared = f"MC-3 closed: MC-3a's recorded outcome at {closure[:9]}, NOTES-C1 §5.11"

    def rec(system, data, ratio, **kw):
        r = dict(kind="md2_run", system=system, data=data, md2_cleared=cleared, commit=head,
                 calib_file_sha256=REGISTERED_CALIB, calib_check=dict(ok=True), dirty=False,
                 registered_check=dict(same=True, p90_same=True),
                 lib_sha256="a" * 64, host_lib_sha256="b" * 64, host_api=13,
                 driver_sha256=DRIVER_SHA256,
                 protocol=dict(reps=REPS, attempts=ATTEMPTS, warmup_s=WARMUP_S,
                               primer_mib=PRIMER_MIB, seed=SEED, max_spread_pct=MAX_SPREAD),
                 levels={str(N): dict(num_moduli=N, g=N) | BAR_CONFIG for N in LEVELS},
                 bitexact_packed_vs_inline={str(N): True for N in LEVELS}, contaminated=False,
                 # medians: S14 = ratio, F14 = 1 exactly (x2 and x0.5 are exact)
                 reps=[{"S14": {"total_ms": ratio * f}, "F14": {"total_ms": f}}
                       for f in (1.0, 2.0, 0.5)])
        r.update(kw)
        return r

    def both(system, rn, rd):
        return [rec(system, "normal", rn), rec(system, "decay_opp", rd)]

    def all_pass():
        return [r for s in GATED for r in both(s, SYSTEMS[s]["bar"], SYSTEMS[s]["bar"])]

    def outcome(recs, system, data):
        return decide(recs)[1].get((system, data), {}).get("outcome")

    # the registered constants, pinned as literals (a change to any is a re-registration)
    check("bars and p90", [(SYSTEMS[s]["bar"], SYSTEMS[s]["p90"]) for s in GATED],
          [(0.750, 0.15), (0.664, 0.15), (0.673, 0.15)])
    check("protocol constants", (LEVELS, MAX_SPREAD, ATTEMPTS, WARMUP_S, REPS, PHI, SEED,
                                 PRIMER_MIB, MAX_RECORDS, CAP, DATA, GATED, MD2_SYSTEMS),
          ((6, 8, 9, 10, 14), 5.0, 2, 3.0, 15, 0.5, 1, 1024, 2, 0.8, ("normal", "decay_opp"),
           ("ppl", "df", "sq4096"), ("ppl", "df", "sq4096", "jacobi")))
    check("the runs", RUNS, {
        "S14": ([(5, 6), (4, 8), (4, 10), (3, 14)], False), "F14": ([(16, 14)], False),
        "S9": ([(6, 6), (10, 9)], False), "F9": ([(16, 9)], False), "F8": ([(16, 8)], False),
        "P14": ([(5, 6), (4, 8), (4, 10), (3, 14)], True), "PF14": ([(16, 14)], True)})
    # the reviewer's reproduction: one data set twice is not both data sets (MD2-3)
    recs = [rec("sq4096", "normal", 0.70), rec("sq4096", "normal", 0.70)]
    roles, pairs, systems, md2 = decide(recs)
    check("same data set twice: sq4096", systems["sq4096"], "INCOMPLETE")
    check("same data set twice: M-d2", md2, "INCOMPLETE")
    check("same data set twice: the second's role", roles[1][0],
          "after the decisive record, not considered")
    check("same data set twice: decay_opp", pairs.get(("sq4096", "decay_opp")), None)
    # all three gated systems inside on both data sets; the Jacobi view is reported only
    check("all inside", decide(all_pass())[3], "PASS")
    check("all inside, Jacobi outside", decide(all_pass() + both("jacobi", 2.0, 2.0))[3], "PASS")
    check("Jacobi outside: reported", outcome(all_pass() + both("jacobi", 2.0, 2.0), "jacobi",
                                              "normal"), "OUTSIDE")
    # one data set outside fails the system and M-d2
    recs = [r for s in ("ppl", "df") for r in both(s, SYSTEMS[s]["bar"], SYSTEMS[s]["bar"])]
    recs += both("sq4096", 0.673, 0.80)
    _, _, systems, md2 = decide(recs)
    check("decay_opp outside: sq4096", systems["sq4096"], "FAIL")
    check("decay_opp outside: M-d2", md2, "FAIL")
    check("an outside pair decides FAIL with another pair missing",
          decide([rec("df", "normal", 0.9)])[2]["df"], "FAIL")
    # band edges: relative band, cap 0.8 on the gated three; every band pinned as literals
    b = band("ppl")
    check("ppl band", (b["lo"], b["hi"]), (0.750 * (1 - 0.15), 0.8))
    for r_, want in ((b["lo"], "INSIDE"), (math.nextafter(b["lo"], 0), "OUTSIDE"),
                     (0.8, "INSIDE"), (math.nextafter(0.8, 1), "OUTSIDE")):
        check(f"ppl at {r_!r}", outcome([rec("ppl", "normal", r_)], "ppl", "normal"), want)
    b = band("df")
    check("df band", (b["lo"], b["hi"]), (0.664 * (1 - 0.15), 0.664 * (1 + 0.15)))
    b = band("sq4096")
    check("sq4096 band", (b["lo"], b["hi"]), (0.673 * (1 - 0.15), 0.673 * (1 + 0.15)))
    check("the bands, rounded", [tuple(round(band(s)[k], 4) for k in ("lo", "hi"))
                                 for s in GATED],
          [(0.6375, 0.8), (0.5644, 0.7636), (0.5721, 0.774)])
    b = band("jacobi")
    check("jacobi band (reported)", (b["lo"], b["hi"], b["gated"]),
          (0.786 * (1 - 0.30), 0.786 * (1 + 0.30), False))
    # rule (a): contaminated or not bit-exact is inadmissible, never PASS or FAIL
    bad = rec("df", "normal", 0.664, contaminated=True)
    check("contaminated inside alone", outcome([bad], "df", "normal"), "INCOMPLETE")
    check("contaminated outside alone", outcome([rec("df", "normal", 0.9, contaminated=True)],
                                                "df", "normal"), "INCOMPLETE")
    check("contaminated, then inside", outcome([bad, rec("df", "normal", 0.664)], "df",
                                               "normal"), "INSIDE")
    check("contaminated, then outside", outcome([bad, rec("df", "normal", 0.9)], "df",
                                                "normal"), "OUTSIDE")
    check("contaminated missing", outcome([rec("df", "normal", 0.664, contaminated=None)], "df",
                                          "normal"), "INCOMPLETE")
    nobe = rec("df", "normal", 0.664, bitexact_packed_vs_inline={"6": True})
    check("bit-exact check missing: inadmissible", outcome([nobe], "df", "normal"), "INCOMPLETE")
    check("bit-exact check missing: not a defect", bitexact_failed(nobe), [])
    # a bit-exact value that is not literally true is inadmissible, and not a defect
    for v in (None, 0, 1, "yes", "true"):
        r_ = rec("df", "normal", 0.664,
                 bitexact_packed_vs_inline={str(N): True for N in LEVELS} | {"10": v})
        check(f"bit-exact {v!r} at a level: inadmissible", outcome([r_], "df", "normal"),
              "INCOMPLETE")
        check(f"bit-exact {v!r} at a level: not STOPPED", decide(all_pass() + [r_])[3], "PASS")
    # rule (c): the first admissible record decides; a later one never replaces it
    check("outside, then inside", outcome([rec("df", "normal", 0.9), rec("df", "normal", 0.664)],
                                          "df", "normal"), "OUTSIDE")
    check("inside, then outside", outcome([rec("df", "normal", 0.664), rec("df", "normal", 0.9)],
                                          "df", "normal"), "INSIDE")
    # rule (b)/(d): one re-run; both inadmissible -> INCOMPLETE; a third is excess
    recs = [bad, bad, rec("df", "normal", 0.664)]
    roles, pairs, _, _ = decide(recs)
    check("two inadmissible, then a third", pairs[("df", "normal")]["outcome"], "INCOMPLETE")
    check("two inadmissible: the cause", pairs[("df", "normal")].get("why", "").startswith("both"), True)
    check("two inadmissible: no automatic continuation",
          "reviewer verdict or Eric's decision" in pairs[("df", "normal")].get("why", ""), True)
    check("the third's role", roles[2][0], "excess: past the one re-run, not considered")
    check("one inadmissible: re-run allowed",
          decide([bad])[1][("df", "normal")].get("why", "").startswith("no admissible"), True)
    # a bit-exact failure on any record in the file stops M-d2, whatever else holds
    nbe = {str(N): N != 10 for N in LEVELS}
    check("bit-exact failure (Jacobi view)", decide(all_pass() + [
        rec("jacobi", "normal", 0.786, bitexact_packed_vs_inline=nbe)])[3], "STOPPED")
    check("bit-exact failure (excess record)", decide(all_pass() + [
        rec("df", "normal", 0.664), rec("df", "normal", 0.664, bitexact_packed_vs_inline=nbe)
    ])[3], "STOPPED")
    check("bit-exact failure on a smoke record", decide(all_pass() + [
        rec("smoke", "normal", 0.86, bitexact_packed_vs_inline=nbe)])[3], "STOPPED")
    check("bit-exact failure: the levels", bitexact_failed(rec("df", "normal", 0.6,
                                                               bitexact_packed_vs_inline=nbe)),
          [10])
    check("bit-exact failure: inadmissible", outcome(
        [rec("df", "normal", 0.664, bitexact_packed_vs_inline=nbe)], "df", "normal"),
        "INCOMPLETE")
    why = decide([rec("df", "normal", 0.664, bitexact_packed_vs_inline=nbe)])[1][
        ("df", "normal")].get("why", "")
    check("STOPPED: the failing pair offers no re-run",
          (why.startswith("M-d2 STOPPED"), "one re-run allowed" in why), (True, False))
    # what `run` refuses (rules (a)-(c))
    check("run: first record", rerun_problems([], "df", "normal"), [])
    check("run: after an inadmissible one", rerun_problems([bad], "df", "normal"), [])
    check("run: other data set", rerun_problems([rec("df", "normal", 0.6)], "df", "decay_opp"),
          [])
    check("run: decided", len(rerun_problems([rec("df", "normal", 0.9)], "df", "normal")), 1)
    check("run: re-run used", len(rerun_problems([bad, bad], "df", "normal")), 1)
    check("run: stopped", len(rerun_problems(
        [rec("ppl", "normal", 0.7, bitexact_packed_vs_inline=nbe)], "df", "normal")), 1)
    check("run: stopped by a smoke record", len(rerun_problems(
        [rec("smoke_t", "normal", 0.7, bitexact_packed_vs_inline=nbe)], "df", "normal")), 1)
    check("run: a value that is not true is no stop", rerun_problems(
        [rec("ppl", "normal", 0.7, bitexact_packed_vs_inline={"6": None})], "df", "normal"), [])
    # provenance: a clean record passes; each departure is refused
    check("provenance: clean", provenance_problems(rec("ppl", "normal", 0.75), anc), [])
    check("file: clean", file_problems(all_pass() + both("jacobi", 0.786, 0.786), anc), [])
    lv = {str(N): dict(num_moduli=N, g=N) | BAR_CONFIG for N in LEVELS}
    for what, kw in (
            ("dirty", dict(dirty=True)), ("dirty unknown", dict(dirty=None)),
            ("no reason", dict(md2_cleared="")), ("blank reason", dict(md2_cleared="  ")),
            ("no reason (None)", dict(md2_cleared=None)),
            ("a reason with no citation", dict(md2_cleared="MC-3 closed")),
            ("no NOTES-C1 section", dict(md2_cleared=f"MC-3 closed at {closure[:9]}")),
            ("no commit", dict(md2_cleared="MC-3 closed, NOTES-C1 §5.11")),
            ("a commit that is no ancestor", dict(md2_cleared="MC-3 closed at abcdef0, "
                                                              "NOTES-C1 §5.11")),
            ("the closure after the run", dict(commit="2" * 40)),
            ("no commit recorded", dict(commit=None)),
            ("another driver", dict(driver_sha256="0" * 64)),
            ("another calibration file", dict(calib_file_sha256="0" * 64)),
            ("library calibration differs", dict(calib_check=dict(ok=False))),
            ("no calibration check", dict(calib_check=None)),
            ("bar not reproduced", dict(registered_check=dict(same=False, p90_same=True))),
            ("p90 class not reproduced", dict(registered_check=dict(same=True, p90_same=False))),
            ("no registered check", dict(registered_check=None)),
            ("reps", dict(protocol=dict(rec("ppl", "normal", 0.75)["protocol"], reps=5))),
            ("seed", dict(protocol=dict(rec("ppl", "normal", 0.75)["protocol"], seed=2))),
            ("g != N", dict(levels=lv | {"9": dict(lv["9"], g=3)})),
            ("num_moduli != N", dict(levels=lv | {"9": dict(lv["9"], num_moduli=10)})),
            ("RESIDENT", dict(levels=lv | {"6": dict(lv["6"], vec_mode="RESIDENT")})),
            ("packed B", dict(levels=lv | {"14": dict(lv["14"], packed="B")})),
            ("TRUNC", dict(levels=lv | {"8": dict(lv["8"], rounding="TRUNC")})),
            ("a level missing", dict(levels={k: v for k, v in lv.items() if k != "10"})),
            ("old record (no levels config)", dict(levels={str(N): dict(N=N, g=N,
                                                                        vec_mode="GENERATE")
                                                           for N in LEVELS}))):
        check(f"provenance refuses: {what}",
              bool(provenance_problems(rec("ppl", "normal", 0.75, **kw), anc)), True)
    check("provenance: 'no calib_check' named",
          provenance_problems(rec("ppl", "normal", 0.75, calib_check=None), anc),
          ["no calib_check"])
    # one library per file: the M-d2 records share (lib_sha256, host_lib_sha256, host_api)
    for what, kw in (("another library", dict(lib_sha256="c" * 64)),
                     ("another host library", dict(host_lib_sha256="c" * 64)),
                     ("another host API", dict(host_api=12)),
                     ("no library recorded", dict(lib_sha256=None))):
        recs = all_pass()[:-1] + [rec("sq4096", "decay_opp", 0.673, **kw)]
        check(f"file refused: {what}", bool(file_problems(recs, anc)), True)
    check("file: a smoke record of another library is no M-d2 record",
          file_problems(all_pass() + [rec("smoke", "normal", 0.86, lib_sha256="c" * 64,
                                          md2_cleared=None)], anc), [])
    # the NaN-filled bit-exact check (numpy only here; skipped if it is not installed)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        v = np.array([1.5, -2.0, 3.25, 0.0]).view(np.uint64)
        nanv = np.full(4, NAN_BITS, dtype=np.uint64)
        check("NaN pattern is a NaN", bool(np.isnan(nanv.view(np.float64)).all()), True)
        check("bit-exact: equal and fully written", bitexact_level(v, v.copy()), (True, 0, 0))
        check("bit-exact: a differing bit", bitexact_level(v, v ^ np.uint64(1))[0], False)
        check("bit-exact: nothing written by either call (vacuous)",
              bitexact_level(nanv, nanv.copy()), (False, 4, 4))
        check("bit-exact: the packed call wrote nothing", bitexact_level(v, nanv), (False, 0, 4))
        w = v.copy()
        w[2] = NAN_BITS
        check("bit-exact: one element unwritten by both", bitexact_level(w, w.copy()),
              (False, 1, 1))
        check("bit-exact: a NaN result of another payload is a result",
              bitexact_level(np.array([float("nan")]).view(np.uint64),
                             np.array([float("nan")]).view(np.uint64))[0], True)
    print(f"md2_time selftest: {n[0] - len(fails)} of {n[0]} checks passed")
    for f in fails:
        print("  FAIL " + f)
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    host_default = os.path.join(ROOT, "build", "libimmerdink_host.so")
    p = sub.add_parser("predict", help="the [M] predictions, bars and bands (CPU)")
    p.add_argument("--host-lib", default=host_default)
    p.add_argument("--out")
    r = sub.add_parser("run", help="time one system on the GPU (one fresh process)")
    r.add_argument("system", choices=sorted(SYSTEMS))
    r.add_argument("--data", choices=DATA, default="normal")
    r.add_argument("--lib", default=os.path.join(ROOT, "build", "libimmerdink.so"))
    r.add_argument("--out", required=True)
    r.add_argument("--reps", type=int, default=REPS)
    r.add_argument("--attempts", type=int, default=ATTEMPTS)
    r.add_argument("--warmup-s", type=float, default=WARMUP_S)
    r.add_argument("--primer-mib", type=int, default=PRIMER_MIB)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--md2-cleared", default=None,
                   help="why M-d2's ordering (NOTES-C1 §5.9) is met, citing the commit and the "
                   "NOTES-C1 section that record MC-3's closure; required for the M-d2 systems")
    j = sub.add_parser("judge", help="evaluate run records against the bands (CPU)")
    j.add_argument("records")
    j.add_argument("--host-lib", default=host_default)
    j.add_argument("--out")
    sub.add_parser("selftest", help="the decision and provenance rules on synthetic records "
                   "(CPU, standard library)")
    a = ap.parse_args()
    return {"predict": cmd_predict, "run": cmd_run, "judge": cmd_judge,
            "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
