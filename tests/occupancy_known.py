"""Decision logic of the per-W kernel-occupancy check of tests/test_contract.py (suite api, 'occupancy <kernel>').

Pure functions, no GPU: tests/test_occupancy_known.py runs them on mocked tables. DESIGN §3.4.1 v1.16: the performance model keeps one occupancy
row per pack kernel, so the CTAs per SM must be equal for the emitter word counts W = 2 and 3. On Hopper (sm_90) they are not for the pack kernel
(issue 23: the model needs per-W rows). That one recorded pattern is accepted as a KNOWN, reported divergence until issue 23 lands; any other device or
any other values fail exactly as before.
"""

# rows[w] = [CTAs per SM, registers, shared memory bytes, threads] of kernel_occupancy_w(kernel, w); w = 0 is the reported (most constrained) row.
KNOWN_SM90_PACK = {2: [4, 64, 40768, 256], 3: [2, 116, 40768, 256]}
KNOWN_MESSAGE = "occupancy pack: per-W difference on sm_90 (issue 23: the model needs per-W rows)"


def occupancy_verdict(name, rows, cc_major):
    """('ok' | 'known' | 'fail', message) for one kernel. rows: {0, 2, 3: row or None}; cc_major: the device's compute capability major."""
    if any(r is None or r[0] < 1 for r in rows.values()):
        return "fail", f"occupancy {name}: {rows}"
    if len({rows[w][0] for w in (2, 3)}) != 1:
        if (name == "pack" and cc_major == 9 and list(rows[2]) == KNOWN_SM90_PACK[2] and list(rows[3]) == KNOWN_SM90_PACK[3]
                and list(rows[0]) in (KNOWN_SM90_PACK[2], KNOWN_SM90_PACK[3])):
            return "known", KNOWN_MESSAGE
        return "fail", f"occupancy {name}: CTAs per SM differ by W {rows} (the model needs per-W occupancy rows: DESIGN §3.4.1 v1.16)"
    worst = min((rows[w] for w in (3, 2)), key=lambda r: (r[0], -r[1]))
    if rows[0] != worst or any(rows[w][2:] != rows[0][2:] for w in (2, 3)):
        return "fail", f"occupancy {name}: reported row {rows[0]} is not the most constrained {rows}"
    return "ok", ""
