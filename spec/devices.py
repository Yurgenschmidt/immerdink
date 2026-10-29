"""Reference model of the device-class rules (docs/NOTES-18.md D3-D5), independent of the C++.

    select(device, rows)            -> (match, table_id, row_index)
    code_kind(err, binary, ptx, major, minor)
    plan_status(match, code)

tests/test_devices.py compares these with libimmerdink_host on a grid of synthetic cards, and
checks that each mutated variant of the rule (MUTANTS) is told apart from the real one by that grid.
A device is a dict {cc_major, cc_minor, sm_count, name}; a row a dict {table_id, cc_major, cc_minor,
sm_count, name_prefix}.
"""

EXACT, ARCH, GENERATION, DEFAULT, UNSUPPORTED = range(5)
NATIVE, COMPATIBLE, RECOMPILED, UNUSABLE = range(4)
P_NATIVE, P_APPROXIMATE, P_RECOMPILED, P_UNSUPPORTED = range(4)

# The built-in rows (src/host/device_class.cpp kRows).
ROWS = [dict(table_id=0, cc_major=8, cc_minor=9, sm_count=128, name_prefix="NVIDIA GeForce RTX 4090"),
        dict(table_id=1, cc_major=9, cc_minor=0, sm_count=132, name_prefix="NVIDIA H100")]


def select(d, rows=ROWS, mutant=None):
    cc = 10 * d["cc_major"] + d["cc_minor"]
    min_cc = 90 if mutant == "min_cc_90" else 80
    if cc < min_cc or not rows:
        return UNSUPPORTED, -1, -1
    same_cc = [i for i, w in enumerate(rows) if (w["cc_major"], w["cc_minor"]) == (d["cc_major"], d["cc_minor"])]

    def name_ok(w):
        return True if mutant == "ignore_name" else d["name"].startswith(w["name_prefix"])

    def sm_ok(w):
        return True if mutant == "ignore_sms" else w["sm_count"] in (0, d["sm_count"])

    exact = [i for i in same_cc if sm_ok(rows[i]) and name_ok(rows[i])]
    if mutant == "arch_before_exact":
        if same_cc:
            return ARCH, rows[same_cc[0]]["table_id"], same_cc[0]
    if exact:
        return EXACT, rows[exact[0]]["table_id"], exact[0]
    if same_cc:
        return ARCH, rows[same_cc[0]]["table_id"], same_cc[0]
    gen = [i for i, w in enumerate(rows) if w["cc_major"] == d["cc_major"]]
    if mutant == "generation_any_major":
        gen = list(range(len(rows)))
    if gen:
        def key(i):
            dist = abs(rows[i]["cc_minor"] - d["cc_minor"])
            if mutant == "ties_upward":
                return (dist, -rows[i]["cc_minor"], i)
            return (dist, rows[i]["cc_minor"], i)
        best = min(gen, key=key)
        return GENERATION, rows[best]["table_id"], best
    de = [i for i, w in enumerate(rows) if (w["cc_major"], w["cc_minor"]) == (8, 9)]
    i = de[0] if de else 0
    return DEFAULT, rows[i]["table_id"], i


def code_kind(err, binary, ptx, major, minor, mutant=None):
    if err != 0:
        return UNUSABLE
    dev = 10 * major + minor
    if binary == ptx:
        if binary == dev:
            return NATIVE
        if binary < dev and binary // 10 == major:
            return COMPATIBLE
        if mutant == "compat_any_major" and binary < dev:
            return COMPATIBLE
        return UNUSABLE
    if mutant == "recompiled_any_binary":
        return RECOMPILED if ptx < dev else UNUSABLE
    return RECOMPILED if binary == dev and ptx < dev else UNUSABLE


def plan_status(match, code, mutant=None):
    if mutant == "recompiled_before_unsupported" and code == RECOMPILED:
        return P_RECOMPILED
    if match == UNSUPPORTED or code == UNUSABLE:
        return P_UNSUPPORTED
    if mutant == "approx_over_recompiled" and (match != EXACT):
        return P_APPROXIMATE
    if code == RECOMPILED:
        return P_RECOMPILED
    if match != EXACT or code == COMPATIBLE:
        return P_APPROXIMATE
    return P_NATIVE


MUTANTS = ("min_cc_90", "ignore_name", "ignore_sms", "arch_before_exact", "generation_any_major",
           "ties_upward", "compat_any_major", "approx_over_recompiled", "recompiled_any_binary", "recompiled_before_unsupported")
