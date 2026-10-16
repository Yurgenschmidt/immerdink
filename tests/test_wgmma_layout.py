#!/usr/bin/env python3
"""The layouts the Hopper K4 design depends on (issue 17, docs/NOTES-17.md), read from CUTLASS's CuTe.

    python3 tests/test_wgmma_layout.py --cutlass third_party/cutlass --include CUDA_INCLUDE [--cxx g++]

CPU only. Compiles tests/wgmma_layout_dump.cpp (the layout tables of CUTLASS 4.8's SM90 int8 wgmma atoms and
shared-memory swizzles), then checks:

 1. WGMMA ACCUMULATOR. Thread t of a warpgroup (warp w = t / 32, lane l = t % 32), register v of an
    m64nNk32 s32 <- s8 x s8 instruction holds, with g = l / 4, q = l % 4:
        row = 16 w + g + 8 ((v / 2) % 2),   column = 8 (v / 4) + 2 q + (v % 2).
    Every atom CuTe defines (N = 8 16 32 64 96 128 192 256) follows it, and it is a bijection onto the
    64 x N tile for every multiple of 8 up to 256.
 2. K4's EPILOGUE. K4 (mma.sync) keeps its accumulators in CUTLASS's MmaTensorOp order, element (16 mi + 8 h + g,
    8 nj + 2 q + e) of a warp tile at index 4 (mi + nj kMi) + 2 h + e, and its epilogue (src/kernels/k4_gemm.cuh)
    transposes 4 x 4 words inside each lane quad so that lane q ends with the 16 consecutive bytes
    16 q .. 16 q + 15 of the row. With wgmma the rows of one instruction are the same g, h pattern per warp
    (kMi = 1) and the column blocks nj run through the register index v = 4 nj + 2 h + e, so the epilogue
    applies per 64-column group c with registers 32 c + (4 nj + 2 h + e), unchanged. Checked here by running a
    Python copy of the epilogue's data movement (byte packing, quad transpose, store) on wgmma-ordered registers
    from the dump: every thread ends up with its row's bytes 16 q .. 16 q + 15, for every N that is a multiple
    of 64 (the epilogue's group width).
 3. SHARED-MEMORY SWIZZLE of the K-major int8 operand tiles. CuTe's SW128 / SW64 / SW32 atoms are
        lin = r W + b;  offset(r, b) = lin xor ((((lin >> 7) mod (W / 16))) << 4)     (W = 128, 64, 32 bytes;
    CuTe's Swizzle<log2(W/16), 4, 3>: the 16-byte chunk index is xor-ed with the next bits of the address),
    bijections of the atom. For SW128 (K tile = 128 bytes: one atom wide), the eight rows of an atom put the
    same 16-byte chunk in eight different 16-byte bank groups (conflict-free column access), which is what
    TMA's SWIZZLE_128B writes and the wgmma descriptor expects.
"""
import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


class Fail(Exception):
    pass


def build_and_run(cutlass, include, cxx):
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "dump")
        cmd = [cxx, "-std=c++17", "-O1", "-I", os.path.join(cutlass, "include"), *sum([["-I", i] for i in include], []),
               os.path.join(HERE, "wgmma_layout_dump.cpp"), "-o", exe]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise Fail("cannot compile wgmma_layout_dump.cpp:\n" + r.stderr[-3000:])
        out = subprocess.run([exe], capture_output=True, text=True, check=True).stdout
    acc, swz = {}, {}
    for line in out.splitlines():
        f = line.split()
        if f[0] == "acc":
            n, t, v, idx = map(int, f[1:])
            acc.setdefault(n, {})[(t, v)] = (idx % 64, idx // 64)
        elif f[0] == "swz":
            swz.setdefault(f[1], {})[(int(f[2]), int(f[3]))] = int(f[4])
        else:
            raise Fail("unexpected dump line: " + line)
    return acc, swz


def wg_pos(t, v):
    w, l = divmod(t, 32)
    g, q = divmod(l, 4)
    return 16 * w + g + 8 * ((v // 2) % 2), 8 * (v // 4) + 2 * q + (v % 2)


def check_accumulator(acc):
    for n, m in sorted(acc.items()):
        if len(m) != 128 * (n // 2):
            raise Fail(f"N={n}: {len(m)} entries")
        for (t, v), rc in m.items():
            if wg_pos(t, v) != rc:
                raise Fail(f"N={n} thread {t} register {v}: CuTe {rc}, formula {wg_pos(t, v)}")
    for n in range(8, 257, 8):  # the PTX shape range: the formula is a bijection onto 64 x N
        seen = {wg_pos(t, v) for t in range(128) for v in range(n // 2)}
        if seen != {(r, c) for r in range(64) for c in range(n)}:
            raise Fail(f"N={n}: formula is not a bijection onto the tile")
    return len(acc)


def byte_perm(a, b, sel):
    """CUDA __byte_perm(a, b, sel) on 32-bit words."""
    src = [(a >> (8 * i)) & 255 for i in range(4)] + [(b >> (8 * i)) & 255 for i in range(4)]
    r = 0
    for i in range(4):
        r |= src[(sel >> (4 * i)) & 7] << (8 * i)
    return r


def quad_transpose(lanes):
    """lanes: 4 lists of 4 words (lane t = quad lane index); k4_gemm.cuh quad_transpose."""
    for s in (1, 2):
        for k in range(4):
            if k & s:
                continue
            send = [lanes[t][k] if t & s else lanes[t][k ^ s] for t in range(4)]
            recv = [send[t ^ s] for t in range(4)]
            for t in range(4):
                if t & s:
                    lanes[t][k] = recv[t]
                else:
                    lanes[t][k ^ s] = recv[t]


def epilogue_bytes(vals):
    """k4_gemm.cuh epilogue_rows for one row of one quad: vals[q][nj][e] = residue (a byte) held by lane q at
    column block nj (8 blocks = 64 columns), element e. Returns, per lane, the 16 bytes it stores."""
    w = []
    for q in range(4):
        wq = []
        for k in range(4):
            qq = [vals[q][2 * k + (e >> 1)][e & 1] for e in range(4)]
            wq.append(byte_perm(byte_perm(qq[0], qq[1], 0x0040), byte_perm(qq[2], qq[3], 0x0040), 0x5410))
        w.append(wq)
    quad_transpose(w)
    out = []
    for q in range(4):
        x = [byte_perm(w[q][0], w[q][1], 0x5410), byte_perm(w[q][2], w[q][3], 0x5410),
             byte_perm(w[q][0], w[q][1], 0x7632), byte_perm(w[q][2], w[q][3], 0x7632)]
        out.append([(word >> (8 * i)) & 255 for word in x for i in range(4)])
    return out


def check_epilogue(acc):
    n_checked = 0
    for n, m in sorted(acc.items()):
        if n % 64:
            continue
        # a value that names its element: byte = (row * 7 + col * 13) % 251 + 1  (distinct enough per row)
        def val(r, c):
            return (r * 7 + c * 13) % 251 + 1
        # registers by (thread, v), in wgmma order, from the dump
        reg = {(t, v): val(*rc) for (t, v), rc in m.items()}
        for c0 in range(0, n // 64):               # 64-column groups
            for w in range(4):                      # warps of the warpgroup
                for g in range(8):
                    for h in range(2):              # rows g and g + 8 of the warp's 16
                        row = 16 * w + g + 8 * h
                        vals = []
                        for q in range(4):
                            t = 32 * w + 4 * g + q
                            vals.append([[reg[(t, 32 * c0 + 4 * nj + 2 * h + e)] for e in range(2)] for nj in range(8)])
                        out = epilogue_bytes(vals)
                        for q in range(4):
                            want = [val(row, 64 * c0 + 16 * q + j) for j in range(16)]
                            if out[q] != want:
                                raise Fail(f"N={n} group {c0} warp {w} row {row}: lane {q} stores {out[q]}, want {want}")
        n_checked += 1
    return n_checked


def check_swizzle(swz):
    for kind, W in (("SW128", 128), ("SW64", 64), ("SW32", 32)):
        m = swz[kind]
        for (r, b), off in m.items():
            lin = r * W + b
            want = lin ^ (((lin >> 7) & (W // 16 - 1)) << 4)   # Swizzle<log2(W/16), 4, 3>
            if off != want:
                raise Fail(f"{kind} ({r},{b}): CuTe {off}, formula {want}")
        if sorted(m.values()) != list(range(8 * W)):
            raise Fail(f"{kind} is not a bijection onto the atom")
    m = swz["SW128"]
    for chunk in range(8):   # eight rows, same logical 16-byte chunk: eight different bank groups
        slots = {(m[(r, 16 * chunk)] % 128) // 16 for r in range(8)}
        if len(slots) != 8:
            raise Fail(f"SW128 chunk {chunk}: rows share bank groups {slots}")
    return 3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cutlass", default=os.path.join(os.path.dirname(HERE), "third_party", "cutlass"))
    ap.add_argument("--include", action="append", default=[], help="CUDA runtime header directories (a ;-list is split; an include/cccl next to one is added)")
    ap.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    a = ap.parse_args()
    try:
        incs = [d for x in a.include for d in x.split(";") if d]   # CMake passes a list
        incs += [os.path.join(d, "cccl") for d in incs if os.path.isdir(os.path.join(d, "cccl"))]
        acc, swz = build_and_run(a.cutlass, list(dict.fromkeys(incs)), a.cxx)
        n1 = check_accumulator(acc)
        n2 = check_epilogue(acc)
        n3 = check_swizzle(swz)
    except Fail as e:
        print("FAIL  " + str(e))
        return 1
    print(f"PASS  {n1} CuTe accumulator atoms match the formula (and it is a bijection for N = 8..256 step 8); "
          f"K4's epilogue data movement stores the right bytes for {n2} wgmma widths; {n3} swizzle atoms match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
