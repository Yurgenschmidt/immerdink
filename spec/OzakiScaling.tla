---------------------------- MODULE OzakiScaling ----------------------------
(***************************************************************************)
(* Fast-mode scaling, quantization, unscaling and error bound of the       *)
(* Ozaki-II pipeline, over a toy floating-point format (module ToyFloat).  *)
(*                                                                         *)
(* DESIGN.md map                                                           *)
(*   §3.2 exact-integer shift rule  -> RowInfo (S, gamma, u, sigma),       *)
(*        claims GammaIsIlogb (steps 1-2), SExact, SBounded (step 3),      *)
(*        ShiftIsMaximal (step 4), SigmaInRange, ShiftFits (step 5)        *)
(*   §3.2 "Guarantee"               -> NormBudget, CRTPrecondition         *)
(*   §3.2 plan-time RN rejection    -> ASSUME DesignRules; models          *)
(*        pos_kappa_boundary, pos_kappa_2Tm1, neg_kappa_2T, neg_noreject   *)
(*   §3.3 RN / TRUNC quantization   -> RowInfo.q, QuantFormat,             *)
(*        QuantIsRNEven / QuantIsTrunc (bit-exact rounding rule, against  *)
(*        an independent dyadic reference)                                 *)
(*   §3.4 residues of mu*2^eps      -> ResidueFactorization                *)
(*   §3.6 centered CRT range        -> CRTRoundTrip                        *)
(*   §3.7 single final rounding     -> FinalRounding (ToyFloat!RoundOnce   *)
(*        checked against the reference IsCorrectRN, incl. subnormal       *)
(*        results, -0 and overflow to Inf)                                 *)
(*   §3.8 error contract            -> ErrorBound (l1 form, exact),        *)
(*        SigmaBound (the 2^-sigma step of the l2 form), ContractL2 (the  *)
(*        l2 contract end to end, about D, incl. the u|D| term)            *)
(*   non-vacuity                    -> No* witness sentinels, expected to  *)
(*        FAIL in neg_witness_* models: ||a'|| = T and sum|a'||b'| = H in  *)
(*        BOTH modes, the RN kappa slack used, quantizer ties, final-      *)
(*        rounding ties / carry-out / carry into Inf / subnormal / -0 /    *)
(*        underflow / Inf, and ContractL2's sqrt branch                    *)
(*                                                                         *)
(* One state = one input: a row a of A (row models) or a row a of A and a  *)
(* column b of B (joint models). The behaviour picks an input and         *)
(* stutters, so every invariant is checked over ALL admissible inputs.    *)
(* Row claims are symmetric in the entry order, so rows are enumerated as *)
(* sorted sequences (multisets); joint claims are symmetric under         *)
(* permuting the (a_h, b_h) pairs, so pair lists are sorted instead.      *)
(*                                                                         *)
(* SCALING (toy -> real widths; spec/params.py certifies the real ones):  *)
(*   Prec/EMin/EMax  toy format (FP64: 53 / -1022 / 1023), with subnormals *)
(*   F               fixed-point bits of S (real 60). Toy F = 2..3 makes   *)
(*                   the ceiling in S visibly lossy; F = 2(Prec-1) or more *)
(*                   makes every term exact (as at real F = 60 >= 2*52),   *)
(*                   which is what lets rows sit ON the RN budget          *)
(*                   (pos_row_rn_tight, pos_kappa_floor_ref).              *)
(*   Moduli, P       toy moduli, P = product; H = (P-1) \div 2 and         *)
(*                   T = floor(sqrt H) exactly as DESIGN §2. H itself is   *)
(*                   pinned by pos_joint_rn_P18 (P/2 = 9 is a square, so   *)
(*                   H = P div 2 would give T = 3 and break CRTRoundTrip); *)
(*                   OzakiCRT's CRTExactBeyond pins it on the CRT side.    *)
(*   K               contraction length k (real up to 2^66 for S < 2^128); *)
(*                   Kappa = ceil(sqrt K). floor(sqrt K) is UNSOUND        *)
(*                   (neg_kappa_floor at the non-square K = 2).            *)
(*   NZ              number of enumerated entries; the other K - NZ        *)
(*                   entries are 0. Zeros add nothing to S, a', z, the    *)
(*                   norms or the error, and only loosen the K term of the *)
(*                   bound, so NZ < K is exact padding (used only where K  *)
(*                   must be large to reach a Kappa boundary).             *)
(*   ShiftBits       signed width that stores sigma (real: int16)         *)
(* Every width and constant enters only through these parameters; the     *)
(* rule, the quantizer and the rounding are written exactly as for the    *)
(* real widths.                                                            *)
(*                                                                         *)
(* Mutation (negative models only) removes one design rule:              *)
(*   "none"             the design                                         *)
(*   "noKappa"          RN shift rule without the kappa slack              *)
(*   "floorS"           floor instead of ceiling in S                      *)
(*   "Tplus1"           shift rule uses T + 1                              *)
(*   "rnQuantTruncRule" RN quantization with the TRUNC shift rule (same   *)
(*                      inequality as noKappa: 4S4^u <= (2T)^2 2^F)        *)
(*   "noReject"         no plan-time rejection of kappa >= T (RN)          *)
(*   "doubleRound"      final rounding to Prec bits, then to the subnormal *)
(*                      grid ([G8])                                        *)
(*   "kappaFloor"       shift rule with kappa = floor(sqrt k) (isqrt slip) *)
(*   "quantTiesAway"    RN quantizer rounds ties away from zero (C round() *)
(*                      instead of rint())                                 *)
(***************************************************************************)
EXTENDS Integers, Sequences, ToyFloat

CONSTANTS
    F,          \* fixed-point bits of the sum of squares S (real: 60)
    Moduli,     \* <<p_1, ..., p_N>>, pairwise coprime
    K,          \* contraction length k
    NZ,         \* enumerated entries per row (the rest are zero), NZ <= K
    Mode,       \* "RN" (default) or "TRUNC"
    Mutation,   \* "none" in positive models; see header
    ShiftBits   \* signed bits available to store sigma (real: 16)

VARIABLES a, b, ra, rb
vars == <<a, b, ra, rb>>

-----------------------------------------------------------------------------
(* Helpers *)

RECURSIVE SumF(_, _)
SumF(f, n) == IF n = 0 THEN 0 ELSE f[n] + SumF(f, n - 1)

RECURSIVE ProdSeq(_)
ProdSeq(s) == IF s = <<>> THEN 1 ELSE Head(s) * ProdSeq(Tail(s))

SMod(x, m) == LET r == x % m IN IF r >= (m + 1) \div 2 THEN r - m ELSE r
Coprime(x, y) == \A d \in 2..Min(x, y) : ~(x % d = 0 /\ y % d = 0)

Hs == 1..NZ

\* All sorted (non-decreasing) length-n sequences over the integer set S.
RECURSIVE Sorted(_, _)
Sorted(n, S) == IF n = 0 THEN {<<>>}
                ELSE UNION {{<<v>> \o s : s \in Sorted(n - 1, {w \in S : w >= v})}
                            : v \in S}

-----------------------------------------------------------------------------
(* Derived constants (zero-arity, so TLC evaluates them once) *)

P     == ProdSeq(Moduli)
H     == (P - 1) \div 2                                       \* DESIGN §2
T     == CHOOSE t \in 0..H : t * t <= H /\ (t + 1) * (t + 1) > H
Kappa == CHOOSE c \in 1..K : (c - 1) * (c - 1) < K /\ K <= c * c  \* ceil sqrt
KappaFloor == CHOOSE c \in 0..K : c * c <= K /\ K < (c + 1) * (c + 1)  \* mutation

\* The shift rule as the implementation evaluates it (mutations apply).
TR       == IF Mutation = "Tplus1" THEN T + 1 ELSE T
KappaR   == CASE Mutation = "noKappa"    -> 0
              [] Mutation = "kappaFloor" -> KappaFloor
              [] OTHER                   -> Kappa
RuleMode == IF Mutation = "rnQuantTruncRule" THEN "TRUNC" ELSE Mode
LhsC     == IF RuleMode = "RN" THEN 4 ELSE 1
\* RN: 4*S*2^(2u) <= (2T - kappa)^2 * 2^F ; TRUNC: S*2^(2u) <= T^2 * 2^F
Rhs      == IF RuleMode = "RN" THEN (2 * TR - KappaR) * (2 * TR - KappaR) * 2^F
            ELSE TR * TR * 2^F

Rule(S, u) == IF u >= 0 THEN LhsC * S * 4^u <= Rhs
              ELSE LhsC * S <= Rhs * 4^(-u)

\* The largest u with Rule(S, u), computed as an implementation would:
\* with d = BitLen(Rhs) - BitLen(LhsC*S), log2(Rhs/(LhsC*S)) lies in
\* (d-1, d+1), so the maximum is c or c-1 for c = floor((d+1)/2): one
\* clz pair and ONE comparison. ShiftIsMaximal checks this algorithm.
\* The division MUST be a FLOOR division (TLC's \div; in C an arithmetic
\* shift (d + 1) >> 1): d is negative at real widths when k is near the
\* RN maximum (N = 2..4, k = (T-1)^2 give d = -4), and C's truncating '/'
\* then yields c - 1 = -2 where the maximum is -3 or -2. For u < 0 the
\* comparison is LhsC*S <= Rhs * 4^-u. Models whose d < 0 guard this:
\* pos_kappa_2Tm1 and pos_trunc_smallT.
UOf(S) == LET c == (BitLen(Rhs) - BitLen(LhsC * S) + 1) \div 2
          IN  IF Rule(S, c) THEN c ELSE c - 1

-----------------------------------------------------------------------------
(* §3.2 + §3.3 for one row (or column) r: a function Hs -> Vals, values   *)
(* in fixed point (units 2^Q). Implementation-level integer arithmetic.   *)

QuantMode == Mode

RowInfo(r) ==
  LET mx    == CHOOSE m \in {Abs(r[h]) : h \in Hs} : \A h \in Hs : Abs(r[h]) <= m
      nz    == mx # 0
      gamma == IF nz THEN ILogb(mx) ELSE 0            \* step 1
      \* step 2-3: x_h = X_h * 2^(Q - gamma); term = ceil(x_h^2 * 2^F)
      e2    == 2 * (Q - gamma) + F
      Term(X) == IF e2 >= 0 THEN X * X * 2^e2
                 ELSE IF Mutation = "floorS" THEN FloorDiv(X * X, 2^(-e2))
                 ELSE CeilDiv(X * X, 2^(-e2))
      S     == SumF([h \in Hs |-> Term(r[h])], NZ)
      u     == IF S = 0 THEN 0 ELSE UOf(S)            \* step 4
      sigma == IF S = 0 THEN 0 ELSE u - gamma
      \* §3.3: a'_h = RN_even(a_h * 2^sigma) or trunc(a_h * 2^sigma)
      sh    == Q + sigma
      Quant(X) == IF sh >= 0 THEN X * 2^sh
                  ELSE IF QuantMode = "RN" THEN
                       IF Mutation = "quantTiesAway"
                       THEN Sgn(X) * ((Abs(X) + 2^(-sh - 1)) \div 2^(-sh))
                       ELSE RNEDiv(X, 2^(-sh))
                  ELSE TruncDiv(X, 2^(-sh))
      q     == [h \in Hs |-> Quant(r[h])]
  IN  [S |-> S, gamma |-> gamma, e2 |-> e2, u |-> u, sigma |-> sigma, q |-> q,
       nnz |-> SumF([h \in Hs |-> IF r[h] = 0 THEN 0 ELSE 1], NZ),
       n2  |-> SumF([h \in Hs |-> r[h] * r[h]], NZ),     \* ||a||^2 = n2 * 4^Q
       l1  |-> SumF([h \in Hs |-> Abs(r[h])], NZ),       \* ||a||_1 = l1 * 2^Q
       qq  |-> SumF([h \in Hs |-> q[h] * q[h]], NZ)]     \* ||a'||^2

-----------------------------------------------------------------------------
(* Closed-form range of sigma (the [cert] formula; params.py evaluates    *)
(* the same expressions at Prec = 53, EMin = -1022, EMax = 1023, F = 60). *)
(* Every nonzero row has S >= 2^F (its max entry has |x| >= 1) and       *)
(* S <= K * ceil(xmax^2 * 2^F), xmax = (2^Prec - 1) / 2^(Prec-1); gamma   *)
(* ranges over Q..EMax.                                                    *)

SMaxRow  == K * CeilDiv((2^Prec - 1) * (2^Prec - 1) * 2^F, 4^(Prec - 1))
SigmaMax == UOf(2^F) - Q
SigmaMin == UOf(SMaxRow) - EMax
ShiftLo  == -(2^(ShiftBits - 1))
ShiftHi  == 2^(ShiftBits - 1) - 1

-----------------------------------------------------------------------------
(* Assumptions *)

Modes     == {"RN", "TRUNC"}
Mutations == {"none", "noKappa", "floorS", "Tplus1", "rnQuantTruncRule",
              "noReject", "doubleRound", "kappaFloor", "quantTiesAway"}

WellFormed ==
    /\ Prec >= 2 /\ EMin <= EMax /\ F >= 0
    /\ K >= 1 /\ NZ \in 1..K /\ ShiftBits >= 1
    /\ Mode \in Modes /\ Mutation \in Mutations
    /\ Len(Moduli) >= 1 /\ \A l \in 1..Len(Moduli) : Moduli[l] >= 2
    /\ \A i, j \in 1..Len(Moduli) : i # j => Coprime(Moduli[i], Moduli[j])
    /\ T >= 1

\* DESIGN §3.2 step 4: plan time rejects kappa >= T in RN mode.
DesignRules == Mutation # "noReject" => (Mode = "RN" => Kappa < T)

ASSUME WellFormed
ASSUME DesignRules

\* The closed-form sigma range is attained (all rows enumerated, design).
ASSUME (NZ = K /\ Mutation = "none") =>
         /\ RowInfo([h \in Hs |-> IF h = 1 THEN 1 ELSE 0]).sigma = SigmaMax
         /\ RowInfo([h \in Hs |-> MaxX]).sigma = SigmaMin

-----------------------------------------------------------------------------
(* Inputs *)

Zeros == [h \in Hs |-> 0]
\* Operators with an argument: TLC eagerly evaluates every zero-arity constant
\* definition at startup, and the joint pair lists are huge at row-model sizes.
Rows(n) == Sorted(n, Vals)

\* (a_h, b_h) pairs encoded as integers so that pair lists can be sorted.
W           == 2 * MaxX + 1
Enc(x, y)   == (x + MaxX) * W + (y + MaxX)
DecA(c)     == c \div W - MaxX
DecB(c)     == (c % W) - MaxX
PairLists(n) == Sorted(n, {Enc(x, y) : x \in Vals, y \in Vals})

RowInit == /\ a \in Rows(NZ)
           /\ b = Zeros
           /\ ra = RowInfo(a)
           /\ rb = RowInfo(b)

JointInit == \E pl \in PairLists(NZ) :
           /\ a = [h \in Hs |-> DecA(pl[h])]
           /\ b = [h \in Hs |-> DecB(pl[h])]
           /\ ra = RowInfo(a)
           /\ rb = RowInfo(b)

Next == UNCHANGED vars
RowSpec   == RowInit   /\ [][Next]_vars
JointSpec == JointInit /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Row invariants (checked for a, and for b in joint models) *)

TypeOK == /\ a \in [Hs -> Vals] /\ b \in [Hs -> Vals]
          /\ ra = RowInfo(a) /\ rb = RowInfo(b)

\* §3.2 steps 1-2: gamma = ilogb(max_h |a_h|), stated without ILogb, BitLen
\* or the row maximum of RowInfo: every |a_h| < 2^(gamma+1) and some
\* |a_h| >= 2^gamma. Equivalently (step 2) the normalized x_h = a_h 2^-gamma
\* satisfy |x_h| < 2 and max_h |x_h| >= 1 (subnormal maxima included).
GammaIsIlogbOf(r, ri) == ri.nnz = 0 \/
    /\ \A h \in Hs : DLt(<<Abs(r[h]), Q>>, <<1, ri.gamma + 1>>)
    /\ \E h \in Hs : DLeq(<<1, ri.gamma>>, <<Abs(r[h]), Q>>)
GammaIsIlogb == GammaIsIlogbOf(a, ra) /\ GammaIsIlogbOf(b, rb)

\* §3.2 step 3: S is the exact integer ceil-sum, i.e.
\*   ||x||^2 * 2^F <= S < ||x||^2 * 2^F + (#nonzero terms)
\* where ||x||^2 * 2^F = n2 * 2^(2(Q - gamma) + F) (x_h = X_h 2^(Q-gamma)).
\* The exponent is recomputed from gamma and F here (not RowInfo's e2), so
\* a wrong fixed-point width in S is caught.
SExactOf(ri) == ri.S = 0 \/
    LET nx == <<ri.n2, 2 * (Q - ri.gamma) + F>>
    IN  /\ DLeq(nx, DInt(ri.S))
        /\ DLt(DInt(ri.S), DAdd(nx, DInt(ri.nnz)))
SExact == SExactOf(ra) /\ SExactOf(rb)

\* §3.2 step 3 / [cert] S < 2^128: per row, 2^F <= S <= SMaxRow
\* = K * ceil(xmax^2 2^F), xmax = (2^Prec - 1)/2^(Prec-1) (the toy analog
\* of "S fits uint128 for k < 2^66"; params.py checks the real one).
SBoundedOf(ri) == ri.nnz = 0 \/ (2^F <= ri.S /\ ri.S <= SMaxRow)
SBounded == SBoundedOf(ra) /\ SBoundedOf(rb)

\* §3.2 step 4: u satisfies the rule and u + 1 does not. Rule(S, _) is
\* antitone in u (S > 0), so u is THE largest integer satisfying it.
ShiftIsMaximalOf(ri) == ri.S = 0 \/ (Rule(ri.S, ri.u) /\ ~Rule(ri.S, ri.u + 1))
ShiftIsMaximal == ShiftIsMaximalOf(ra) /\ ShiftIsMaximalOf(rb)

\* §3.2 Guarantee: ||a'||_2 <= T (RN via the kappa/2 slack, TRUNC directly).
\* Uses the TRUE T, so the Tplus1 mutation is caught.
NormBudget == ra.qq <= T * T /\ rb.qq <= T * T

\* §3.3: a' = mu * 2^eps with |mu| <= 2^Prec and eps >= 0 (never needs
\* an int64 container; real: |mu| <= 2^53).
QuantFormatOf(ri) == \A h \in Hs :
    \E e \in 0..BitLen(Abs(ri.q[h])) :
        ri.q[h] % 2^e = 0 /\ Abs(ri.q[h]) \div 2^e <= 2^Prec
QuantFormat == QuantFormatOf(ra) /\ QuantFormatOf(rb)

\* §3.4: residues via r_l = SMod((mu mod p) * (2^eps mod p), p), with the
\* mathematical (non-negative) mod, for the canonical mu*2^eps split.
ResidueFactorizationOf(ri) == \A h \in Hs, l \in 1..Len(Moduli) :
    LET x == ri.q[h]
        e == CHOOSE e \in 0..BitLen(Abs(x)) :
                 x % 2^e = 0 /\ Abs(x) \div 2^e <= 2^Prec
        mu == x \div 2^e
        p == Moduli[l]
    IN  SMod((mu % p) * (2^e % p), p) = SMod(x, p)
ResidueFactorization == ResidueFactorizationOf(ra) /\ ResidueFactorizationOf(rb)

\* §3.3 bit-exactly (DESIGN §1 bitwise determinism: the oracle must match
\* every tie). Stated against the exact dyadic v_h = a_h 2^sigma
\* = <<X_h, Q + sigma>>, independently of RNEDiv / TruncDiv:
\*   RN:    |a'_h - v_h| <= 1/2, and an exact tie goes to an even a'_h;
\*   TRUNC: |a'_h| <= |v_h| < |a'_h| + 1 and a'_h has the sign of v_h.
QuantIsRNEvenOf(r, ri) == Mode = "RN" => \A h \in Hs :
    LET v  == <<r[h], Q + ri.sigma>>
        d2 == DMul(DAbs(DSub(DInt(ri.q[h]), v)), DInt(2))
    IN  /\ DLeq(d2, DInt(1))
        /\ DEq(d2, DInt(1)) => ri.q[h] % 2 = 0
QuantIsRNEven == QuantIsRNEvenOf(a, ra) /\ QuantIsRNEvenOf(b, rb)
QuantIsTruncOf(r, ri) == Mode = "TRUNC" => \A h \in Hs :
    LET v == <<r[h], Q + ri.sigma>>
    IN  /\ DLeq(DInt(Abs(ri.q[h])), DAbs(v))
        /\ DLt(DAbs(v), DInt(Abs(ri.q[h]) + 1))
        /\ ri.q[h] * r[h] >= 0
QuantIsTrunc == QuantIsTruncOf(a, ra) /\ QuantIsTruncOf(b, rb)

\* §3.2 step 5 / [cert]: sigma lies in the closed-form range ...
SigmaInRangeOf(ri) == ri.S = 0 \/ ri.sigma \in SigmaMin..SigmaMax
SigmaInRange == SigmaInRangeOf(ra) /\ SigmaInRangeOf(rb)
\* ... and fits the signed ShiftBits store (real: int16).
ShiftFits == ra.sigma \in ShiftLo..ShiftHi /\ rb.sigma \in ShiftLo..ShiftHi

\* §3.8: the 2^-sigma step of the l2 contract, as it actually holds:
\*   2^-sigma < 2 ||a||_2 sqrt(1 + K 2^-F) / (T - kappa/2)      (RN)
\*   2^-sigma < 2 ||a||_2 sqrt(1 + K 2^-F) / T                  (TRUNC)
\* squared:  4^-sigma * Den^2 * 2^F < 16 * ||a||^2 * (2^F + K),
\* Den = 2T - kappa (RN) or 2T (TRUNC). The sqrt(1 + K 2^-F) factor is the
\* ceiling slack in S; DESIGN omits it (see SigmaBoundLiteral).
Den == IF Mode = "RN" THEN 2 * T - Kappa ELSE 2 * T
SigmaBoundOf(ri) == ri.S = 0 \/
    DLt(<<Den * Den * 2^F, -2 * ri.sigma>>, <<16 * ri.n2 * (2^F + K), 2 * Q>>)
SigmaBound == SigmaBoundOf(ra) /\ SigmaBoundOf(rb)
\* DESIGN §3.8 as written: 2^-sigma <= 2 ||a||_2 / (T - kappa/2).
SigmaBoundLiteralOf(ri) == ri.S = 0 \/
    DLeq(<<Den * Den, -2 * ri.sigma>>, <<16 * ri.n2, 2 * Q>>)
SigmaBoundLiteral == SigmaBoundLiteralOf(ra) /\ SigmaBoundLiteralOf(rb)

-----------------------------------------------------------------------------
(* Joint invariants (row a of A, column b of B) *)

Z  == SumF([h \in Hs |-> ra.q[h] * rb.q[h]], NZ)   \* z = (A'B')_ij, exact
Sh == ra.sigma + rb.sigma                          \* sigma + tau
V  == <<Z, -Sh>>                                   \* z * 2^-(sigma+tau)
AB == <<SumF([h \in Hs |-> a[h] * b[h]], NZ), 2 * Q>>  \* sum_h a_h b_h, exact

\* §3.2 Guarantee / OzakiCRT's requirement: sum_h |a'_h||b'_h| <= H.
CRTPrecondition == SumF([h \in Hs |-> Abs(ra.q[h]) * Abs(rb.q[h])], NZ) <= H

\* §3.6: the centered CRT range recovers z (SMod(z, P) = z).
CRTRoundTrip == SMod(Z, P) = Z

\* §3.8, l1 form, exact (before the final rounding):
\*  RN:    |2^-(s+t) z - ab| <= 2^-(t+1)|a|_1 + 2^-(s+1)|b|_1 + K 2^-(s+t+2)
\*  TRUNC: |2^-(s+t) z - ab| <= 2^-t    |a|_1 + 2^-s    |b|_1 + K 2^-(s+t)
ErrorBound ==
  LET c  == IF Mode = "RN" THEN 1 ELSE 0
      rhs == DAdd(DAdd(<<ra.l1, Q - rb.sigma - c>>, <<rb.l1, Q - ra.sigma - c>>),
                  <<K, -Sh - 2 * c>>)
  IN  DLeq(DAbs(DSub(V, AB)), rhs)

\* §3.7: D is the correctly rounded (RN-even) toy value of z * 2^-(s+t),
\* including subnormal results, -0, and overflow to +-Inf.
D == IF Mutation = "doubleRound" THEN RoundTwice(V) ELSE RoundOnce(V)
FinalRounding == IsCorrectRN(D, V)

\* §3.8 l2 contract END TO END, about D (as corrected; see design_issues):
\*  RN:    |D - ab| <= 2 sqrt(k) ||a|| ||b|| s / (T - kappa/2)
\*                     + k 2^-(sigma+tau+2) + u|D|
\*  TRUNC: |D - ab| <= 4 sqrt(k) ||a|| ||b|| s / T
\*                     + k 2^-(sigma+tau)   + u|D|
\* with s = sqrt(1 + k 2^-F) and u = 2^-Prec. Guard (DESIGN: "no over/
\* underflow"): D finite and normal, or z = 0 (then D = V = 0 exactly).
\* With L = |D - ab| - (k term) - u|D|, the claim is L <= 0 or
\*   L^2 * Den^2 * 2^F <= c * k * ||a||^2 ||b||^2 * (2^F + k),
\* Den = 2T - kappa (RN, c = 16) or 2T (TRUNC, c = 64); ||a||^2 = n2 4^Q.
\* L is normalized (odd numerator) so the squares stay small for TLC; the
\* right side is a plain 32-bit product, which bounds the joint models'
\* formats (n2a*n2b*c*k*(2^F+k) < 2^31). TLC aborts loudly on overflow,
\* so a too-large model can never pass silently.
RECURSIVE DNorm(_)
DNorm(x) == IF x[1] # 0 /\ x[1] % 2 = 0 THEN DNorm(<<x[1] \div 2, x[2] + 1>>)
            ELSE x
DSigned  == <<IF D.neg THEN -D.X ELSE D.X, Q>>
KTerm    == IF Mode = "RN" THEN <<K, -Sh - 2>> ELSE <<K, -Sh>>
LExcess  == DNorm(DSub(DSub(DAbs(DSub(DSigned, AB)), KTerm), <<D.X, Q - Prec>>))
CC       == IF Mode = "RN" THEN 16 ELSE 64
ContractGuard == ~D.inf /\ (D.X >= 2^(Prec - 1) \/ Z = 0)
\* Contract with the leading term divided by sqrt(m) (m = 1: the claim).
ContractL2Scaled(m) == ContractGuard =>
    LET l == LExcess IN
    l[1] <= 0 \/
    DLeq(<<(l[1] * Den) * (l[1] * Den) * m, 2 * l[2] + F>>,
         <<CC * K * ra.n2 * rb.n2 * (2^F + K), 4 * Q>>)
ContractL2 == ContractL2Scaled(1)
\* NOTE: at toy sizes the leading term has slack: in the RN joint models
\* the excess L exceeds 1/4 of it (NoContractQuarter is violated) but
\* never 1/2 (checked in a scratch run); TRUNC has more than 4x. So dropping DESIGN's missing factor
\* s = sqrt(1 + k 2^-F) is NOT detectable end to end; it is caught one
\* step earlier, by SigmaBound vs SigmaBoundLiteral (neg_sigmabound_literal).

-----------------------------------------------------------------------------
(* Witness (non-vacuity) sentinels. Each is EXPECTED TO BE VIOLATED by a  *)
(* neg_witness_* model run on a positive configuration: it shows the      *)
(* positive models really reach the boundary or the rounding case.        *)

NoNormAtT      == ra.qq < T * T          \* ||a'||^2 = T^2 is reached
NoCRTAtH       == SumF([h \in Hs |-> Abs(ra.q[h]) * Abs(rb.q[h])], NZ) < H
NoInf          == ~D.inf                 \* overflow to +-Inf is reached
NoSubnormalD   == D.inf \/ D.X \notin SubX   \* subnormal D is reached
NoNegZero      == D.inf \/ ~(D.neg /\ D.X = 0)  \* -0 (negative underflow)
NoUnderflowLoss == D.inf \/ D.X # 0 \/ Z = 0  \* nonzero z rounding to 0
\* ContractL2 is not vacuous: the excess error L exceeds a quarter of the
\* leading term (the sqrt branch of ContractL2 is really exercised).
NoContractQuarter == ContractL2Scaled(16)
\* RN shift-rule slack really used: some RN row has ||a'|| > T - kappa/2.
NoKappaSlackUsed == Mode = "RN" => 4 * ra.qq <= (2 * T - Kappa) * (2 * T - Kappa)
\* Final-rounding sub-cases of RoundOnce (V at the target exponent ex):
\* a carry-out to 2^Prec, a carry-out that overflows to Inf, an exact tie.
TargetShift(v) == LET m == Abs(v[1])  E == BitLen(m) - 1 + v[2]
                  IN  v[2] - (Max(E, EMin) - Prec + 1)
CarryOf(v) == v[1] # 0 /\ RNShift(Abs(v[1]), TargetShift(v)) = 2^Prec
TieOf(v)   == v[1] # 0 /\ TargetShift(v) < 0 /\
              2 * (Abs(v[1]) % 2^(-TargetShift(v))) = 2^(-TargetShift(v))
NoCarry    == ~CarryOf(V)
NoCarryOvf == ~(CarryOf(V) /\ D.inf)
NoTie      == ~TieOf(V)
\* Quantization tie (|v_h| exactly halfway) is reached.
NoQuantTie == \A h \in Hs : Q + ra.sigma >= 0 \/
    2 * (Abs(a[h]) % 2^(-(Q + ra.sigma))) # 2^(-(Q + ra.sigma))

=============================================================================
