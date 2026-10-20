"""spec/crt_f64.py against the §3.6 reference (oracle.crt) for every N, at the boundaries of the range and at random points. CPU only.

Any residue vector W (W_l in [0, p_l)) is the image of exactly one z0 in [0, P) (the moduli are coprime), so choosing z0 and setting W_l = z0 mod p_l covers the whole
input space: boundaries (0, 1, H, H + 1, P - 1, the neighbours of multiples that make the quotient S / P nearly an integer) and random z0 are checked.
Also: the bounds of the plan constants hold for every N; a limb width that would make the sums inexact is refused; the downward bias of the quotient estimate is needed
(without it the estimate overshoots for some small z0 and the remainder is negative).

    python3 spec/test_crt_f64.py
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crt_f64 as C  # noqa: E402
import oracle as o  # noqa: E402


def residues(z, p):
    return [z % pl for pl in p]


def main():
    rng = random.Random(21)
    total = 0
    for N in range(o.N_MIN, o.N_MAX + 1):
        c = C.F64Crt(N)
        P, H, p = c.P, c.plan.H, c.plan.p
        zs = {0, 1, 2, 3, 255, 256, 65535, H - 1, H, H + 1, P - 1, P - 2, P - 3, P - 256, P // 2, P // 3, P // 255}
        zs |= set(range(0, 300)) | set(range(P - 300, P))
        zs |= {rng.randrange(0, 1 << rng.randint(1, P.bit_length())) % P for _ in range(1500)}
        zs |= {rng.randrange(P) for _ in range(1500)}
        for z in sorted({x % P for x in zs}):
            W = residues(z, p)
            got = c.crt(W)
            assert got == z, (N, z, got)
            want = o.crt(W, c.plan)
            assert c.crt_centred(W) == want, (N, z)
            total += 1
        # the largest residues: W_l = p_l - 1 (the largest limb sums)
        W = [pl - 1 for pl in p]
        assert c.crt(W) == (-1) % P
        total += 1
    print(f"ok  {total} residue vectors over N = {o.N_MIN}..{o.N_MAX}: crt_f64 equals oracle.crt")

    # limb width: the largest feasible one is chosen, a larger one is refused
    for N in (2, 14, 20):
        c = C.F64Crt(N)
        try:
            C.F64Crt(N, w=c.w + 1)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"w = {c.w + 1} accepted for N = {N}")
    print("ok  an inexact limb width is refused")

    # the bias: without it the quotient estimate overshoots somewhere (z0 small, S / P just above an integer)
    saved = C.EPS_T
    C.EPS_T = 0.0
    bad = 0
    try:
        for N in (14, 16, 20):
            c = C.F64Crt(N)
            P = c.P
            # S / P is just below an integer when z0 is just below P (and just above one when z0 is just above 0)
            for z in [P - 1 - rng.randrange(1 << rng.randint(1, max(2, P.bit_length() - 38))) for _ in range(6000)] + list(range(0, 2000)):
                W = residues(z, c.plan.p)
                s = c.sums(W)
                Sd = s[c.J - 1]
                for j in range(c.J - 2, -1, -1):
                    Sd = math.fma(Sd, float(c.B), s[j])
                S = sum(int(s[j]) << (c.w * j) for j in range(c.J))
                if math.floor(Sd * c.Pinv) > S // P:
                    bad += 1
    finally:
        C.EPS_T = saved
    assert bad > 0, "no overshoot found: the bias would be unneeded (or the search too small)"
    print(f"ok  without the bias the quotient estimate overshoots ({bad} cases found): the bias is needed")
    print("PASS  crt_f64")


if __name__ == "__main__":
    main()
