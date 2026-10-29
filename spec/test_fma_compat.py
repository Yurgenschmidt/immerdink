"""fma_compat.fma against math.fma (Python >= 3.13: bit for bit) and against the IEEE rules it implements. CPU only; python3 spec/test_fma_compat.py"""
import math
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fma_compat as F


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


VALS = [0.0, -0.0, 1.0, -1.0, 3.0, 5e-324, -5e-324, 1.7976931348623157e308, -1.7976931348623157e308, math.inf, -math.inf, math.nan,
        2.0 ** -537, 2.0 ** 537, 1.0000000000000002]


def enc(f):
    try:
        r = f()
    except OverflowError:
        return "O"
    except ValueError:
        return "V"
    return "N" if math.isnan(r) else struct.pack("<d", r).hex()


def check_table():
    """The shim against results of native math.fma (Python 3.14) on a committed grid of edge values, so the check means something on 3.12 too."""
    import itertools
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fma_expected.txt")) as f:
        want = " ".join(l for l in f.read().splitlines() if not l.startswith("#")).split()
    got = [enc(lambda a=a, b=b, c=c: F.fma(a, b, c)) for a, b, c in itertools.product(VALS, repeat=3)]
    assert len(got) == len(want), (len(got), len(want))
    bad = [(i, g, w) for i, (g, w) in enumerate(zip(got, want)) if g != w]
    assert not bad, f"{len(bad)} of {len(want)} differ from native math.fma, first: {bad[:3]}"
    return len(want)


def main():
    ntab = check_table()
    rng = random.Random(7)
    pool = [0.0, -0.0, 1.0, -1.0, 0.25, 0.3, 2.0 ** -1074, -(2.0 ** -1074), 2.0 ** -1022, 2.0 ** 1023, 1.7976931348623157e308,
            math.pi, 1e-300, 3e-300, 2.0 ** -537, 2.0 ** 537]
    n = 0
    ref = math.fma if math.fma is not F.fma else None
    for _ in range(20000):
        a, b, c = (rng.choice(pool) if rng.random() < 0.4 else rng.uniform(-1, 1) * 2.0 ** rng.randint(-1074, 1000) for _ in range(3))
        try:
            got = F.fma(a, b, c)
        except OverflowError:
            got = "overflow"
        if ref is not None:
            try:
                want = ref(a, b, c)
            except OverflowError:
                want = "overflow"
            assert (got == want == "overflow") or (got != "overflow" and want != "overflow" and bits(got) == bits(want)), (a, b, c, got, want)
        n += 1
    # exact, single rounding: (1 + 2^-52) * (1 - 2^-52) - 1 = -2^-104, which a separate multiply and add loses
    a, b = 1.0 + 2.0 ** -52, 1.0 - 2.0 ** -52
    assert F.fma(a, b, -1.0) == -(2.0 ** -104) and a * b - 1.0 == 0.0
    # signs of zero and the non-finite cases
    assert bits(F.fma(-0.0, 1.0, -0.0)) == bits(-0.0) and bits(F.fma(0.25, -0.0, 0.0)) == bits(0.0) and bits(F.fma(1.0, 1.0, -1.0)) == bits(0.0)
    assert F.fma(math.inf, 2.0, 1.0) == math.inf and math.isnan(F.fma(math.nan, 1.0, 1.0))
    for bad in ((0.0, math.inf, 1.0), (math.inf, 1.0, -math.inf)):
        try:
            F.fma(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    try:
        F.fma(1e308, 10.0, 0.0)
    except OverflowError:
        pass
    else:
        raise AssertionError("overflow")
    print(f"PASS  fma_compat: {ntab} grid cases against native results{', ' + str(n) + ' random cases against math.fma' if ref else ''}, exactness, zero signs, non-finite, overflow")


if __name__ == "__main__":
    main()
