"""math.fma for Python before 3.13 (the PACE toolchain env has 3.12): the correctly rounded a * b + c.

The exact value is computed with fractions.Fraction and rounded once by float(), which rounds to nearest even; the sign of a zero result and the
non-finite cases follow IEEE 754 fusedMultiplyAdd; like math.fma, a finite result that overflows raises OverflowError and an invalid operation
(0 * inf + c, or inf - inf) raises ValueError.
"""
import math
from fractions import Fraction


def fma(a, b, c):
    a, b, c = float(a), float(b), float(c)
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
        if math.isnan(a) or math.isnan(b) or math.isnan(c):
            return math.nan
        if (math.isinf(a) and b == 0.0) or (math.isinf(b) and a == 0.0):
            raise ValueError("invalid operation in fma(): 0 * inf")
        if math.isfinite(a) and math.isfinite(b):
            return c             # the exact product is finite (even if a * b overflows in double), c is an infinity
        r = a * b + c            # one rounding is irrelevant here: an infinity or a NaN
        if math.isnan(r):
            raise ValueError("invalid operation in fma(): inf - inf")
        return r
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact == 0:
        px = math.copysign(1.0, a) * math.copysign(1.0, b)
        if a * b == 0.0 and c == 0.0 and px < 0 and math.copysign(1.0, c) < 0:
            return -0.0          # (-0) + (-0)
        return 0.0
    try:
        return float(exact)
    except OverflowError:
        raise OverflowError("overflow in fma()") from None


# Importing this module installs the function as math.fma where Python has none (spec/test_oracle.py relies on it); a program that tests hasattr(math, "fma")
# afterwards sees it, which is the point.
if not hasattr(math, "fma"):
    math.fma = fma
