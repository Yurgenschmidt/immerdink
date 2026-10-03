------------------------------ MODULE ToyFloat ------------------------------
(***************************************************************************)
(* A toy binary floating-point format with subnormals and +-Inf, plus     *)
(* exact dyadic arithmetic, all in TLC's (32-bit, overflow-checked)       *)
(* integers. Helper for OzakiScaling (DESIGN §3.2, §3.3, §3.7, §3.8).      *)
(*                                                                         *)
(* FORMAT (IEEE-754 shaped, scaled down from binary64):                    *)
(*   Prec  significand bits incl. the hidden bit     (binary64: 53)        *)
(*   EMin  exponent of the smallest normal           (binary64: -1022)     *)
(*   EMax  exponent of the largest finite binade     (binary64: 1023)      *)
(* A finite value is M * 2^(e-Prec+1) with                                *)
(*   normal:    e \in EMin..EMax, 2^(Prec-1) <= M < 2^Prec                 *)
(*   subnormal: e = EMin,         1 <= M < 2^(Prec-1)                      *)
(* plus zero. Every finite value is an integer multiple of the quantum    *)
(* 2^Q, Q = EMin - Prec + 1 (binary64: 2^-1074).                           *)
(*                                                                         *)
(* DYADIC FIXED-POINT EMBEDDING: a finite value is represented by the     *)
(* integer X = value * 2^-Q (exact). Products of two values are integers  *)
(* at 2^(2Q), so every exact dot product is an integer too.               *)
(*                                                                         *)
(* DYADICS: <<n, e>> denotes n * 2^e (n, e integers). They are used for   *)
(* the exact rationals of §3.7/§3.8, all of which have power-of-two        *)
(* denominators. Comparison (DLeq) first compares binary magnitudes, so    *)
(* it never shifts a number further than its own bit length (no int32      *)
(* overflow from wildly different exponents).                              *)
(***************************************************************************)
EXTENDS Integers

CONSTANTS Prec, EMin, EMax

Abs(x)    == IF x < 0 THEN -x ELSE x
Sgn(x)    == IF x < 0 THEN -1 ELSE IF x = 0 THEN 0 ELSE 1
Max(x, y) == IF x > y THEN x ELSE y
Min(x, y) == IF x < y THEN x ELSE y

\* ---- integer division helpers, d > 0 (TLC's \div is floor division).
FloorDiv(n, d) == n \div d
CeilDiv(n, d)  == -((-n) \div d)
TruncDiv(n, d) == IF n >= 0 THEN n \div d ELSE -((-n) \div d)
\* Round n/d to nearest, ties to even.
RNEDiv(n, d) == LET q == n \div d
                    r == n - q * d                  \* 0 <= r < d
                IN  IF 2 * r < d THEN q
                    ELSE IF 2 * r > d THEN q + 1
                    ELSE IF q % 2 = 0 THEN q ELSE q + 1

\* Number of significant bits of n >= 0 (BitLen(0) = 0); clz-style.
RECURSIVE BitLen(_)
BitLen(n) == IF n = 0 THEN 0 ELSE 1 + BitLen(n \div 2)

\* x * 2^e as an integer, for e >= 0 or when 2^-e divides x.
Shl(x, e) == IF e >= 0 THEN x * 2^e ELSE x \div 2^(-e)

-----------------------------------------------------------------------------
(* The format *)

Q    == EMin - Prec + 1                       \* quantum exponent
\* Positive finite values in fixed point (units of 2^Q).
SubX == 1..(2^(Prec - 1) - 1)
NormX == {M * 2^(e - EMin) : M \in (2^(Prec - 1))..(2^Prec - 1), e \in EMin..EMax}
PosX == SubX \cup NormX
Vals == {0} \cup PosX \cup {-x : x \in PosX}   \* all finite values
MaxX == (2^Prec - 1) * 2^(EMax - EMin)         \* largest finite value

\* ilogb of a nonzero fixed-point value (subnormals: their true exponent,
\* as C ilogb() returns it).
ILogb(X) == BitLen(Abs(X)) - 1 + Q

\* Significand M of a finite X >= 0, and its parity (for ties-to-even).
UlpUnits(X) == IF X < 2^(Prec - 1) THEN 1 ELSE 2^(BitLen(X) - Prec)
EvenSig(X)  == (X \div UlpUnits(X)) % 2 = 0

-----------------------------------------------------------------------------
(* Exact dyadics <<n, e>> = n * 2^e *)

DZero      == <<0, 0>>
DInt(n)    == <<n, 0>>
DNeg(x)    == <<-x[1], x[2]>>
DAbs(x)    == <<Abs(x[1]), x[2]>>
DMul(x, y) == <<x[1] * y[1], x[2] + y[2]>>
\* n * 2^(e - m) for m <= e.
DAt(x, m)  == x[1] * 2^(x[2] - m)
DAdd(x, y) == LET m == Min(x[2], y[2]) IN <<DAt(x, m) + DAt(y, m), m>>
DSub(x, y) == DAdd(x, DNeg(y))
\* x <= y for x, y >= 0. Magnitude band first: a nonzero <<n,e>> lies in
\* [2^(BitLen(n)-1+e), 2^(BitLen(n)+e)). Equal bands are compared exactly,
\* and then the alignment shift is at most the larger bit length.
PosLeq(x, y) ==
    IF x[1] = 0 THEN TRUE ELSE IF y[1] = 0 THEN FALSE ELSE
    LET lx == BitLen(x[1]) + x[2]
        ly == BitLen(y[1]) + y[2]
    IN  IF lx # ly THEN lx < ly
        ELSE LET m == Min(x[2], y[2]) IN DAt(x, m) <= DAt(y, m)
DLeq(x, y) ==
    LET sx == Sgn(x[1])  sy == Sgn(y[1])
    IN  IF sx # sy THEN sx < sy
        ELSE IF sx >= 0 THEN PosLeq(x, y) ELSE PosLeq(DNeg(y), DNeg(x))
DLt(x, y)  == ~DLeq(y, x)
DEq(x, y)  == DLeq(x, y) /\ DLeq(y, x)
\* Fixed-point value X as a dyadic.
DVal(X)    == <<X, Q>>

-----------------------------------------------------------------------------
(* Rounding an exact dyadic v to the toy format (round to nearest even).   *)
(* Results are records [inf, neg, X]: inf = overflow to +-Inf (X unused), *)
(* neg = sign bit, X = magnitude in fixed point.                           *)

Fin(neg, X) == [inf |-> FALSE, neg |-> neg, X |-> X]
Inf(neg)    == [inf |-> TRUE,  neg |-> neg, X |-> 0]

\* Round m * 2^sh (m >= 0) to an integer, nearest-even (exact if sh >= 0).
RNShift(m, sh) == IF sh >= 0 THEN m * 2^sh ELSE RNEDiv(m, 2^(-sh))

\* DESIGN §3.7: ONE rounding, directly at the final exponent. This is the
\* implementable integer algorithm (bit length -> target exponent,
\* clamped to EMin for subnormals; one RNE shift; carry-out; overflow).
RoundOnce(v) ==
  IF v[1] = 0 THEN Fin(FALSE, 0) ELSE
  LET m   == Abs(v[1])
      E   == BitLen(m) - 1 + v[2]              \* floor(log2 |v|)
      ex  == Max(E, EMin)                      \* subnormal clamp
      M   == RNShift(m, v[2] - (ex - Prec + 1))
      ex2 == IF M = 2^Prec THEN ex + 1 ELSE ex \* carry-out renormalizes
      M2  == IF M = 2^Prec THEN 2^(Prec - 1) ELSE M
  IN  IF ex2 > EMax THEN Inf(v[1] < 0)
      ELSE Fin(v[1] < 0, M2 * 2^(ex2 - EMin))

\* NEGATIVE-MODEL ONLY ([G8] rescale.hpp): round to Prec bits with an
\* unbounded exponent first, then round again onto the subnormal grid.
RoundTwice(v) ==
  IF v[1] = 0 THEN Fin(FALSE, 0) ELSE
  LET m   == Abs(v[1])
      E   == BitLen(m) - 1 + v[2]
      M1  == RNShift(m, v[2] - (E - Prec + 1))
      E1  == IF M1 = 2^Prec THEN E + 1 ELSE E
      N1  == IF M1 = 2^Prec THEN 2^(Prec - 1) ELSE M1
  IN  IF E1 > EMax THEN Inf(v[1] < 0)
      ELSE IF E1 >= EMin THEN Fin(v[1] < 0, N1 * 2^(E1 - EMin))
      ELSE Fin(v[1] < 0, RNEDiv(N1, 2^(EMin - E1)))

\* ---- Reference semantics of RN_even into the format (a SPECIFICATION,  *)
\* ---- not an algorithm): overflow iff |v| >= 2^EMax * (2 - 2^-Prec)     *)
\* ---- (IEEE 754 §7.4: round with unbounded exponent, then overflow);     *)
\* ---- otherwise a nearest finite value, ties to an even significand;    *)
\* ---- the sign bit is the sign of v (a negative v that rounds to 0 is    *)
\* ---- -0; v = 0 is +0).                                                 *)
OvfThreshold == <<2^(Prec + 1) - 1, EMax - Prec>>
Dist(X, w) == DAbs(DSub(DVal(X), w))
IsCorrectRN(D, v) ==
  LET w == DAbs(v) IN
  /\ D.neg = (v[1] < 0)
  /\ D.inf = DLeq(OvfThreshold, w)
  /\ ~D.inf =>
       /\ D.X \in {0} \cup PosX
       /\ \A Y \in {0} \cup PosX :
            LET dD == Dist(D.X, w)  dY == Dist(Y, w)
            IN  \/ DLt(dD, dY)
                \/ DEq(dD, dY) /\ (Y = D.X \/ EvenSig(D.X))

=============================================================================
