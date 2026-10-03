#!/usr/bin/env python3
"""Generate workspace-size test vectors: tensor descriptors -> expected status and bytes.

    python3 tests/gen_workspace_vectors.py OUT

Expected values come from spec/params.py (workspace_size, region_offsets: DESIGN §4a)
applied to the GEMM view produced by `lower` below, a deliberately separate Python
transcription of DESIGN §5 steps 1-8 (partial) and legality rows 1-5. The C++ test
(tests/test_workspace.cpp) compares imd_workspace_size and the C++ lowering against it.

Format: whitespace-separated tokens, one record per case:
    case NAME
    opts N g rounding mode
    tensor A rank e_1..e_r s_1..s_r l_1..l_r      (likewise B, C)
    expect STATUS                                 (when STATUS != SUCCESS)
    expect SUCCESS bytes m n k b swapped
    region NAME start end                         (zero or more, §4a order)
    end
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "spec"))
import params  # noqa: E402

INT64_MAX = 2**63 - 1
INT32_MAX = 2**31 - 1
RN, TRUNC = 0, 1
EMULATE, DIRECT = 0, 1
OK, INVALID, UNSUPPORTED = "SUCCESS", "INVALID_VALUE", "NOT_SUPPORTED"


class Bad(Exception):
    def __init__(self, status):
        super().__init__(status)
        self.status = status


def stride_rule(ext, st):
    """§5 row 4 (spec/IndexMap.tla StrideRule) on a non-empty tensor."""
    big = sorted((abs(s), d, e) for d, (e, s) in enumerate(zip(ext, st)) if e > 1)
    below = 0
    for a, _, e in big:
        if a <= below:
            return False
        below += a * (e - 1)
    return True


def lower(A, B, C):
    """(m, n, k, b, swapped) of DESIGN §5, or raise Bad(status). Tensors are
    (extents, strides, labels) triples."""
    T = (A, B, C)
    for ext, st, lab in T:
        if any(e < 0 for e in ext):
            raise Bad(INVALID)
    ext_of, members, merged = {}, {}, {}
    for x, (ext, st, lab) in enumerate(T):
        for e, s, l in zip(ext, st, lab):
            if ext_of.setdefault(l, e) != e:
                raise Bad(INVALID)                          # row 1
            members.setdefault(l, []).append(x)
            merged[(x, l)] = merged.get((x, l), 0) + s
    if len(C[2]) != len(set(C[2])):
        raise Bad(UNSUPPORTED)                              # row 2
    for l, mem in members.items():
        if set(mem) == {2} and ext_of[l] > 1:
            raise Bad(UNSUPPORTED)                          # row 3
    c_empty = 0 in C[0]
    if not c_empty and not stride_rule(C[0], C[1]):
        raise Bad(INVALID)                                  # row 4
    if any(not -2**63 <= v <= INT64_MAX for v in merged.values()):
        raise Bad(INVALID)                                  # row 5: merged diagonal stride
    for ext, st, lab in T:
        if 0 not in ext and sum(abs(s) * (e - 1) for e, s in zip(ext, st)) > INT64_MAX:
            raise Bad(INVALID)                              # row 5: span
    size = {"I": 1, "J": 1, "K": 1, "L": 1}
    cls = {}
    for l, mem in members.items():
        s = set(mem)
        c = ("L" if s == {0, 1, 2} else "I" if s == {0, 2} else "J" if s == {1, 2}
             else "I" if s == {2} else "K")
        cls[l] = c
        if ext_of[l] != 1:
            size[c] *= ext_of[l]
    if any(v > INT64_MAX for v in size.values()):
        raise Bad(INVALID)                                  # row 5: extent product
    m, n, k, b = size["I"], size["J"], size["K"], size["L"]
    swapped = False
    if not c_empty:
        kept = [(abs(s), d) for d, (e, s) in enumerate(zip(C[0], C[1])) if e > 1]
        if kept:
            _, d = min(kept)
            swapped = cls[C[2][d]] == "I"
    if swapped:
        m, n = n, m
    if max(b * m * k, b * k * n, b * m * n) > INT64_MAX:
        raise Bad(INVALID)                                  # row 5: operand sizes
    if m > INT32_MAX or params.align_up(n, 16) > INT32_MAX:
        raise Bad(INVALID)                                  # row 5: GemmCoord is int
    return m, n, k, b, swapped


MODE_NAME = {EMULATE: params.EMULATE, DIRECT: params.DIRECT}


def expect(A, B, C, N, g, rnd, mode):
    """(status, bytes, (m, n, k, b, swapped), regions)."""
    if not (2 <= N <= 20) or not (0 <= g <= N) or rnd not in (RN, TRUNC) or mode not in MODE_NAME:
        return INVALID, None, None, None
    try:
        m, n, k, b, sw = lower(A, B, C)
    except Bad as e:
        return e.status, None, None, None
    try:
        ws = params.workspace_size(m, n, k, b, N, g, MODE_NAME[mode])
    except OverflowError:
        return INVALID, None, None, None
    return OK, ws, (m, n, k, b, sw), params.region_offsets(m, n, k, b, N, g, MODE_NAME[mode])


def dense(ext, order=None, sign=None):
    """Dense strides for `ext` with mode order[0] fastest (default: last mode fastest)."""
    r = len(ext)
    order = list(range(r - 1, -1, -1)) if order is None else order
    st, acc = [0] * r, 1
    for d in order:
        st[d] = acc
        acc *= max(ext[d], 1)
    if sign:
        st = [s * g for s, g in zip(st, sign)]
    return st


def T(ext, lab, st=None):
    return (list(ext), list(st) if st is not None else dense(ext), list(lab))


# ---------------------------------------------------------------------------------------
# Cases

CASES = []


def add(name, A, B, C, N=14, g=0, rnd=RN, mode=EMULATE, want=None):
    """Record a case. `want` = (status,) or (status, m, n, k, b, swapped) is a hand-derived
    expectation that the generator itself must reproduce."""
    status, ws, mnkb, regs = expect(A, B, C, N, g, rnd, mode)
    if want is not None:
        got = (status,) if len(want) == 1 else (status,) + tuple(mnkb or ())
        assert got == tuple(want), f"{name}: generator gives {got}, hand-derived {want}"
    CASES.append((name, (N, g, rnd, mode), (A, B, C), status, ws, mnkb, regs))


def matrix(m, n, k, b):
    """A[i,k,l] B[k,j,l] C[i,j,l], C dense with j fastest (no role swap unless n = 1)."""
    I, J, K, L = 1, 2, 3, 4
    return (T([m, k, b], [I, K, L]), T([k, n, b], [K, J, L]), T([m, n, b], [I, J, L]))


def grid():
    ms = [0, 1, 15, 16, 17, 100, 257]
    ks = [0, 1, 2, 15, 16, 17, 131056, 131057, 262200]
    for N, g in ((2, 1), (2, 2), (9, 2), (14, 0), (14, 7), (20, 20)):
        for mode in (EMULATE, DIRECT):
            for m in ms:
                for n in ms:
                    for k in ks:
                        for b in (0, 1, 3):
                            add(f"grid_m{m}_n{n}_k{k}_b{b}_N{N}_g{g}_mode{mode}",
                                *matrix(m, n, k, b), N=N, g=g, mode=mode)


def tensors():
    i, j, k, l, x, y, z = 1, 2, 3, 4, 5, 6, 7
    # DESIGN §4a worked examples (also in params.py cert_workspace), as matrices.
    add("worked_17_33_5_3", *matrix(17, 33, 5, 3), N=9, g=2, want=(OK, 17, 33, 5, 3, False))
    add("worked_1000_3_40_2", *matrix(1000, 3, 40, 2), N=20, g=7,
        want=(OK, 1000, 3, 40, 2, False))
    # Diagonals: a label repeated within A or B is summed over its diagonal.
    add("diag_A", T([7, 7, 5], [i, i, k]), T([5, 9], [k, j]), T([7, 9], [i, j]),
        want=(OK, 7, 9, 5, 1, False))
    add("diag_A_K", T([7, 5, 5], [i, k, k]), T([5, 9], [k, j]), T([7, 9], [i, j]),
        want=(OK, 7, 9, 5, 1, False))
    add("diag_B_triple", T([7, 5], [i, k]), T([5, 9, 5, 5], [k, j, k, k]), T([7, 9], [i, j]),
        want=(OK, 7, 9, 5, 1, False))
    add("diag_both_batch", T([3, 4, 3, 6], [l, i, l, k]), T([6, 3, 3, 10], [k, l, l, j]),
        T([4, 3, 10], [i, l, j]), want=(OK, 4, 10, 6, 3, False))
    # Single-operand labels are summed: they join K (stride 0 in the other operand).
    add("A_only", T([7, 5, 3], [i, k, x]), T([5, 9], [k, j]), T([7, 9], [i, j]),
        want=(OK, 7, 9, 15, 1, False))
    add("B_only", T([7, 5], [i, k]), T([4, 5, 9], [y, k, j]), T([7, 9], [i, j]),
        want=(OK, 7, 9, 20, 1, False))
    add("only_singles", T([7, 3], [i, x]), T([4, 9], [y, j]), T([7, 9], [i, j]),
        want=(OK, 7, 9, 12, 1, False))
    add("A_only_extent0", T([7, 5, 0], [i, k, x]), T([5, 9], [k, j]), T([7, 9], [i, j]),
        want=(OK, 7, 9, 0, 1, False))
    # Batch labels, permuted, several per group.
    add("batch_perm", T([3, 6, 4, 2], [l, k, i, z]), T([2, 6, 3, 10], [z, k, l, j]),
        T([10, 3, 4, 2], [j, l, i, z]), want=(OK, 4, 10, 6, 6, False))
    add("batch_perm_swap", T([3, 6, 4, 2], [l, k, i, z]), T([2, 6, 3, 10], [z, k, l, j]),
        T([10, 3, 2, 4], [j, l, z, i]), want=(OK, 10, 4, 6, 6, True))
    add("rank6", T([2, 3, 4, 5, 2, 3], [i, x, k, l, y, z]),
        T([4, 5, 2, 3, 7, 3], [k, l, y, z, j, 9]),
        T([5, 2, 7, 3], [l, i, j, 9]), want=(OK, 2, 21, 4 * 3 * 2 * 3, 5, False))
    # Extent 1: dropped everywhere, including a C-only label and the K group (K-empty).
    add("extent1_conly", T([7, 5], [i, k]), T([5, 9], [k, j]), T([7, 1, 9], [i, x, j]),
        want=(OK, 7, 9, 5, 1, False))
    add("k_empty_outer", T([7], [i]), T([9], [j]), T([7, 9], [i, j]), want=(OK, 7, 9, 1, 1, False))
    add("k_extent1", T([7, 1], [i, k]), T([1, 9], [k, j]), T([7, 9], [i, j]),
        want=(OK, 7, 9, 1, 1, False))
    add("hadamard", T([7, 9], [i, j]), T([7, 9], [i, j]), T([7, 9], [i, j]),
        want=(OK, 1, 1, 1, 63, False))
    add("scalars", T([], []), T([], []), T([], []), want=(OK, 1, 1, 1, 1, False))
    add("dot", T([33], [k]), T([33], [k]), T([], []), want=(OK, 1, 1, 33, 1, False))
    # Zero extents: empty output -> 0 bytes; K extent 0 -> k = 0 -> vec only.
    add("zero_I", T([0, 5], [i, k]), T([5, 9], [k, j]), T([0, 9], [i, j]),
        want=(OK, 0, 9, 5, 1, False))
    add("zero_K", T([7, 0], [i, k]), T([0, 9], [k, j]), T([7, 9], [i, j]),
        want=(OK, 7, 9, 0, 1, False))
    add("zero_L", T([0, 7, 5], [l, i, k]), T([0, 5, 9], [l, k, j]), T([0, 7, 9], [l, i, j]),
        want=(OK, 7, 9, 5, 0, False))
    add("zero_conly", T([7, 5], [i, k]), T([5, 9], [k, j]), T([7, 0, 9], [i, x, j]),
        want=(OK, 0, 9, 5, 1, False))
    add("zero_C_overlapping_strides_ok", T([7, 5], [i, k]), T([5, 0], [k, j]),
        T([7, 0], [i, j], [0, 0]), want=(OK, 7, 0, 5, 1, False))
    # Role swap: C's fastest label is in I -> m and n exchange (n gets the 16-padding).
    add("swap_colmajor_C", T([17, 5], [i, k]), T([5, 33], [k, j]),
        T([17, 33], [i, j], dense([17, 33], [0, 1])), want=(OK, 33, 17, 5, 1, True))
    add("swap_negative_strides", T([17, 5], [i, k]), T([5, 33], [k, j]),
        T([17, 33], [i, j], [-1, -17]), want=(OK, 33, 17, 5, 1, True))
    add("no_swap_negative_strides", T([17, 5], [i, k]), T([5, 33], [k, j]),
        T([17, 33], [i, j], [-33, 1]), want=(OK, 17, 33, 5, 1, False))
    add("swap_batch_fastest_no", T([3, 17, 5], [l, i, k]), T([3, 5, 33], [l, k, j]),
        T([3, 17, 33], [l, i, j], [1, 3, 51]), want=(OK, 17, 33, 5, 3, False))
    # Legality rows and their precedence (§5 table).
    add("row1_extent_mismatch", T([7, 5], [i, k]), T([6, 9], [k, j]), T([7, 9], [i, j]),
        want=(INVALID,))
    add("row1_within_A", T([7, 8], [i, i]), T([9], [j]), T([7, 9], [i, j]), want=(INVALID,))
    add("row1_beats_row2", T([7, 5], [i, k]), T([6, 9], [k, j]), T([7, 7], [i, i]),
        want=(INVALID,))
    add("row2_repeated_C", T([7], [i]), T([7], [i]), T([7, 7], [i, i], [8, 1]),
        want=(UNSUPPORTED,))
    add("row2_repeated_C_extent1", T([1], [i]), T([1], [i]), T([1, 1], [i, i]),
        want=(UNSUPPORTED,))
    add("row2_beats_row4", T([7], [i]), T([7], [i]), T([7, 7], [i, i], [1, 1]),
        want=(UNSUPPORTED,))
    add("row3_conly", T([7, 5], [i, k]), T([5, 9], [k, j]), T([7, 2, 9], [i, x, j]),
        want=(UNSUPPORTED,))
    add("row3_beats_row4", T([7, 5], [i, k]), T([5, 9], [k, j]),
        T([7, 2, 9], [i, x, j], [1, 1, 1]), want=(UNSUPPORTED,))
    add("row4_overlap", T([7, 5], [i, k]), T([5, 9], [k, j]), T([7, 9], [i, j], [1, 1]),
        want=(INVALID,))
    add("row4_zero_stride", T([7, 5], [i, k]), T([5, 9], [k, j]), T([7, 9], [i, j], [0, 1]),
        want=(INVALID,))
    add("row4_tight_ok", T([3, 5], [i, k]), T([5, 4], [k, j]), T([3, 4], [i, j], [4, 1]),
        want=(OK, 3, 4, 5, 1, False))
    add("row4_tight_bad", T([3, 5], [i, k]), T([5, 4], [k, j]), T([3, 4], [i, j], [3, 1]),
        want=(INVALID,))
    add("row4_interleaved_ok", T([3, 5], [i, k]), T([5, 4], [k, j]), T([3, 4], [i, j], [2, 7]),
        want=(OK, 4, 3, 5, 1, True))
    add("negative_extent", T([-1, 5], [i, k]), T([5, 9], [k, j]), T([7, 9], [i, j]),
        want=(INVALID,))
    # Row 5: overflow.
    big = 2**31 - 1
    add("row5_m_int32", *matrix(2**31, 1, 2, 1), want=(INVALID,))
    add("row5_m_int32_ok", *matrix(big, 2, 2, 1), want=(OK, big, 2, 2, 1, False))
    add("row5_nhat_int32", *matrix(2, 2**31 - 15, 2, 1), want=(INVALID,))
    add("row5_nhat_int32_ok", *matrix(2, 2**31 - 16, 2, 1), want=(OK, 2, 2**31 - 16, 2, 1, False))
    add("row5_workspace_u64", *matrix(big, 2**31 - 16, 2, 1), N=20, want=(INVALID,))
    add("row5_workspace_u64_direct", *matrix(big, 2**31 - 16, 2, 1), N=20, mode=DIRECT,
        want=(OK, big, 2**31 - 16, 2, 1, False))
    add("row5_extent_product", T([2**32, 2**32], [i, x]), T([2**32, 2**32], [y, j]),
        T([2**32, 2**32], [i, j]), want=(INVALID,))
    add("row5_product_with_zero", T([2**40, 2**40, 0], [k, x, y]), T([3], [j]), T([3], [j]),
        want=(OK, 1, 3, 0, 1, False))
    add("row5_span", T([7, 3], [i, k], [1, 2**62]), T([3, 9], [k, j]), T([7, 9], [i, j]),
        want=(INVALID,))
    add("row5_span_negative", T([7, 3], [i, k], [1, -2**62]), T([3, 9], [k, j]), T([7, 9], [i, j]),
        want=(INVALID,))
    add("row5_span_edge_ok", T([7, 2], [i, k], [1, 2**63 - 1 - 6]), T([2, 9], [k, j]),
        T([7, 9], [i, j]), want=(OK, 7, 9, 2, 1, False))
    add("row5_merged_diag", T([2, 2], [k, k], [2**62, 2**62]), T([2], [k]), T([], []),
        want=(INVALID,))
    add("int64_min_stride_extent1", T([7, 1], [i, k], [1, -2**63]), T([1, 9], [k, j]),
        T([7, 9], [i, j]), want=(OK, 7, 9, 1, 1, False))
    # Options.
    for name, (N, g, rnd, mode) in (("N1", (1, 0, RN, EMULATE)), ("N21", (21, 0, RN, EMULATE)),
                                    ("g_neg", (9, -1, RN, EMULATE)), ("g_big", (9, 10, RN, EMULATE)),
                                    ("rounding2", (9, 0, 2, EMULATE)), ("mode2", (9, 0, RN, 2))):
        add(f"opt_{name}", *matrix(5, 6, 7, 1), N=N, g=g, rnd=rnd, mode=mode, want=(INVALID,))
    add("opt_trunc_direct_g9", *matrix(5, 6, 7, 2), N=9, g=9, rnd=TRUNC, mode=DIRECT,
        want=(OK, 5, 6, 7, 2, False))


def random_cases(count, seed=20261002):
    rng = random.Random(seed)
    ext_choices = [0, 1, 1, 2, 3, 3, 5, 17]
    for t in range(count):
        nlab = rng.randint(0, 6)
        labs = list(range(10, 10 + nlab))
        ext = {l: rng.choice(ext_choices[1:] if rng.random() < 0.9 else ext_choices) for l in labs}
        modes = []
        for x in range(3):
            r = rng.randint(0, 4)
            pool = labs if labs else []
            modes.append([rng.choice(pool) for _ in range(r)] if pool else [])
        if rng.random() < 0.7:          # usually no repeated label in C
            modes[2] = list(dict.fromkeys(modes[2]))
        tens = []
        for x in range(3):
            e = [ext[l] for l in modes[x]]
            if rng.random() < 0.05 and e:  # extent mismatch
                e[rng.randrange(len(e))] += 1
            order = list(range(len(e)))
            rng.shuffle(order)
            sign = [rng.choice((1, -1)) for _ in e]
            st = dense(e, order, sign)
            if rng.random() < 0.08 and st:  # corrupt a stride
                st[rng.randrange(len(st))] = rng.choice((0, 1, -1, 2))
            tens.append((e, st, modes[x]))
        N = rng.randint(2, 20)
        g = rng.randint(0, N)
        add(f"random_{t}", *tens, N=N, g=g, rnd=rng.choice((RN, TRUNC)),
            mode=rng.choice((EMULATE, EMULATE, DIRECT)))


def write(path):
    out = []
    for name, (N, g, rnd, mode), tens, status, ws, mnkb, regs in CASES:
        out.append(f"case {name}")
        out.append(f"opts {N} {g} {rnd} {mode}")
        for tn, (e, s, l) in zip("ABC", tens):
            out.append(f"tensor {tn} {len(e)} " + " ".join(str(v) for v in e + s + l))
        if status == OK:
            m, n, k, b, sw = mnkb
            out.append(f"expect {status} {ws} {m} {n} {k} {b} {int(sw)}")
            for rn, start, end in regs:
                out.append(f"region {rn} {start} {end}")
        else:
            out.append(f"expect {status}")
        out.append("end")
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    by = {}
    for c in CASES:
        by[c[3]] = by.get(c[3], 0) + 1
    print(f"wrote {len(CASES)} cases to {path}: {by}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    grid()
    tensors()
    random_cases(3000)
    write(sys.argv[1])
