"""Statistics of the campaign kit: paired comparisons, the tie rule and the regression guard.

Standard library only (the cluster job environment carries no numpy at run time of the runner).
Three pieces, all pure functions of timing samples or of summary records:

  median_ci        distribution-free confidence interval of a median (order statistics of the
                   binomial), valid for any sample shape; no normality assumption, robust to outliers
  paired_ratio     the median of the per-pair ratios t_a / t_b of an INTERLEAVED comparison and its
                   interval (ratio > 1: B is faster), computed on log ratios
  classify_pair    the tie rule: inside the interval, or under 10 % on a noisy case, is a tie
  guard            the regression guard: a slowdown beyond max(floor, 2 x spread) is flagged

"spread" is the bench's own definition (bench/imd_bench.cu): (max - min) / median, in percent.
"""
import math

NOISY_SPREAD_PCT = 5.0     # a case whose spread is above this is "noisy" (warm_run.sh BENCH_MAX_SPREAD)
NOISY_MIN_EFFECT = 0.10    # on a noisy case a difference under 10 % is a tie
GUARD_FLOOR = 0.02         # 4090 gate: no case slower than max(2 %, 2 x its spread)
GUARD_K = 2.0
MIN_PAIRS = 6              # below this no 95 % interval of a median exists (2 * 0.5^n > 0.05)


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of nothing")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def spread_pct(xs):
    """(max - min) / median in percent, the bench's `spread_pct`."""
    return 100.0 * (max(xs) - min(xs)) / median(xs)


def _binom_cdf_half(n, k):
    """P(Bin(n, 1/2) <= k), exact for n <= 1100, else the normal approximation with continuity correction."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if n <= 1100:
        return sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    z = (k + 0.5 - n / 2.0) / (0.5 * math.sqrt(n))
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def median_ci(xs, conf=0.95):
    """(lo, hi) with P(lo <= population median <= hi) >= conf, from order statistics; None if n is too small."""
    s = sorted(xs)
    n = len(s)
    alpha = 1.0 - conf
    if n < 2 or 2.0 * 0.5 ** n > alpha and n <= 1100:
        return None
    # the largest j (1-based rank) with P(Bin <= j - 1) <= alpha / 2 gives [x_(j), x_(n + 1 - j)]
    j = 0
    while j < n and _binom_cdf_half(n, j) <= alpha / 2.0:   # cdf(j) = P(<= (j+1) - 1)
        j += 1
    if j < 1:
        return None
    return s[j - 1], s[n - j]


class Paired:
    """A paired comparison: ratio = median over pairs of t_a / t_b (> 1: B faster), [lo, hi] its interval."""

    def __init__(self, ratio, lo, hi, n):
        self.ratio, self.lo, self.hi, self.n = ratio, lo, hi, n

    def __repr__(self):
        return f"Paired(ratio={self.ratio:.4f}, ci=[{self.lo}, {self.hi}], n={self.n})"


def paired_ratio(ta, tb, conf=0.95):
    """The interleaved comparison of two timing series of equal length (pair i = one call of A, one of B)."""
    if len(ta) != len(tb):
        raise ValueError("paired series differ in length")
    if any(not (x > 0.0) or not (y > 0.0) for x, y in zip(ta, tb)):
        raise ValueError("timings must be positive")
    logr = [math.log(x / y) for x, y in zip(ta, tb)]
    ci = median_ci(logr, conf)
    ratio = math.exp(median(logr))
    if ci is None:
        return Paired(ratio, None, None, len(logr))
    return Paired(ratio, math.exp(ci[0]), math.exp(ci[1]), len(logr))


def classify_pair(p, spread=0.0, noisy_spread=NOISY_SPREAD_PCT, noisy_min_effect=NOISY_MIN_EFFECT):
    """'faster' (B beats A), 'slower', 'tie' or 'inconclusive' (too few pairs for an interval).

    The tie rule: the difference is a tie when 1 lies inside the paired interval, or when the case is
    noisy (spread above `noisy_spread` percent) and the median ratio is within `noisy_min_effect` of 1.
    """
    if p.lo is None:
        return "inconclusive"
    if p.lo <= 1.0 <= p.hi:
        return "tie"
    if spread > noisy_spread and abs(p.ratio - 1.0) < noisy_min_effect:
        return "tie"
    return "faster" if p.ratio > 1.0 else "slower"


def guard(base_median, base_spread_pct, new_median, new_spread_pct, floor=GUARD_FLOOR, k=GUARD_K):
    """The regression guard on two summaries. Returns (flagged, slowdown, tolerance).

    slowdown = new / base - 1 (times; positive = slower). The tolerance is max(floor, k x the larger of the
    two spreads), so a noisy case is not flagged for a difference inside its own noise.
    """
    slowdown = new_median / base_median - 1.0
    tol = max(floor, k * max(base_spread_pct, new_spread_pct) / 100.0)
    return slowdown > tol, slowdown, tol


def guard_samples(base, new, floor=GUARD_FLOOR, k=GUARD_K):
    """`guard` on two raw timing series."""
    return guard(median(base), spread_pct(base), median(new), spread_pct(new), floor, k)
