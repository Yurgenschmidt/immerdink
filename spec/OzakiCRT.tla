----------------------------- MODULE OzakiCRT -----------------------------
(***************************************************************************)
(* Integer core of Ozaki scheme II, for ONE output element.               *)
(*                                                                         *)
(* After scaling and truncation (modelled elsewhere), the row a = A'[i,:] *)
(* and the column b = B'[:,j] are integer vectors of length K. The scheme  *)
(* computes their dot product without ever forming it directly:          *)
(*                                                                         *)
(*   1. residues     ra_l[k] = SMod(a[k], m_l)    (must fit ResBits signed)*)
(*   2. int GEMM     acc_l   = sum_k ra_l[k]*rb_l[k] (AccBits accumulator, *)
(*                                                  wraps on overflow)     *)
(*   3. reduce       r_l     = acc_l mod m_l                                *)
(*   4. CRT          C'      = SMod( sum_l w_l * r_l , M )                  *)
(*                                                                         *)
(* Claim (CRTExact): under Preconditions, C' = sum_k a[k]*b[k] exactly.   *)
(*                                                                         *)
(* Every GEMM entry is independent, so one dot product is the whole       *)
(* story for steps 1-4. Matrix/tensor structure is covered by later       *)
(* specs (index maps, schedules). TLC's integers are 32-bit, so the models*)
(* run scaled down (e.g. ResBits = 3, AccBits = 5). The parameters are    *)
(* symbolic, so the same claims are what we instantiate at                *)
(* ResBits = 8, AccBits = 32.                                              *)
(*                                                                         *)
(* Step 4 is specified as EXACT integer arithmetic. How the implementation *)
(* realises it (int128, Garner mixed-radix, or FP64 multi-word) is a       *)
(* refinement of this operator and is an open design decision.            *)
(***************************************************************************)
EXTENDS Integers, Sequences

CONSTANTS
    Moduli,               \* <<m_1, ..., m_N>>, pairwise coprime
    K,                    \* contraction length
    MaxA, MaxB,           \* |A'| <= MaxA, |B'| <= MaxB after scaling/truncation
    ResBits,              \* signed bits per residue operand  (8: INT8 MMA)
    AccBits,              \* signed bits of the MMA accumulator (32: INT32)
    EnforcePreconditions  \* FALSE only in negative (expected-to-fail) models

VARIABLES a, b
vars == <<a, b>>

-----------------------------------------------------------------------------
(* Arithmetic helpers *)

RECURSIVE SumTo(_, _)
SumTo(f, n) == IF n = 0 THEN 0 ELSE f[n] + SumTo(f, n - 1)

RECURSIVE ProdSeq(_)
ProdSeq(s) == IF s = <<>> THEN 1 ELSE Head(s) * ProdSeq(Tail(s))

Abs(x)      == IF x < 0 THEN -x ELSE x
Min(x, y)   == IF x < y THEN x ELSE y
MaxOf(S)    == CHOOSE x \in S : \A y \in S : y <= x

\* Symmetric residue: SMod(x, m) \in -(m \div 2) .. (m - 1) \div 2
\* e.g. m = 256 -> -128..127 (exactly int8), m = 255 -> -127..127.
SMod(x, m)  == LET r == x % m IN IF r >= (m + 1) \div 2 THEN r - m ELSE r

\* Largest |SMod(_, m)|.
Half(m)     == m \div 2

Coprime(x, y) == \A d \in 2..Min(x, y) : ~(x % d = 0 /\ y % d = 0)
Inv(x, m)     == CHOOSE y \in 1..(m - 1) : (x * y) % m = 1

\* Two's-complement wrap of an integer into AccBits signed bits.
Wrap(x)     == SMod(x, 2^AccBits)

-----------------------------------------------------------------------------
(* Derived parameters *)

N       == Len(Moduli)
Ls      == 1..N
Ks      == 1..K
M       == ProdSeq(Moduli)
Mhat(l) == M \div Moduli[l]
W(l)    == Mhat(l) * Inv(Mhat(l) % Moduli[l], Moduli[l])   \* CRT weight w_l

ResLo   == -(2^(ResBits - 1))
ResHi   == 2^(ResBits - 1) - 1
AccMax  == 2^(AccBits - 1) - 1

\* Largest K for which no accumulation order can overflow AccBits.
\* At ResBits = 8, AccBits = 32 with m = 256 present: (2^31-1) \div 128^2 = 131071.
KMax    == AccMax \div MaxOf({Half(Moduli[l]) * Half(Moduli[l]) : l \in Ls})

-----------------------------------------------------------------------------
(* Assumptions: well-formedness always; design preconditions in positive  *)
(* models only.                                                            *)

WellFormed ==
    /\ N >= 1 /\ K >= 1 /\ MaxA >= 0 /\ MaxB >= 0
    /\ \A l \in Ls : Moduli[l] >= 2
    /\ \A i, j \in Ls : i # j => Coprime(Moduli[i], Moduli[j])

Preconditions ==
    /\ \A l \in Ls : Moduli[l] <= 2^ResBits        \* residues fit ResBits
    /\ K <= KMax                                    \* accumulator never overflows
    /\ K * MaxA * MaxB <= (M - 1) \div 2            \* exact result inside SMod range of M

ASSUME WellFormed
ASSUME EnforcePreconditions => Preconditions

-----------------------------------------------------------------------------
(* The pipeline, as pure functions of the inputs *)

Exact     == SumTo([k \in Ks |-> a[k] * b[k]], K)

RA(l)     == [k \in Ks |-> SMod(a[k], Moduli[l])]
RB(l)     == [k \in Ks |-> SMod(b[k], Moduli[l])]
Prods(l)  == [k \in Ks |-> RA(l)[k] * RB(l)[k]]

\* Hardware accumulator. Two's-complement wrap is associative, so the
\* (unspecified) MMA accumulation order cannot change the final value.
Acc(l)    == Wrap(SumTo(Prods(l), K))
R(l)      == Acc(l) % Moduli[l]                    \* in 0 .. m_l - 1

Reconstructed == SMod(SumTo([l \in Ls |-> W(l) * R(l)], N), M)

-----------------------------------------------------------------------------
(* Behaviour: choose any admissible input, then stutter. TLC enumerates   *)
(* every input, so each invariant is checked over all of them.            *)

Init == /\ a \in [Ks -> -MaxA..MaxA]
        /\ b \in [Ks -> -MaxB..MaxB]
Next == UNCHANGED vars
Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Invariants *)

TypeOK      == a \in [Ks -> -MaxA..MaxA] /\ b \in [Ks -> -MaxB..MaxB]

ResiduesFit == \A l \in Ls, k \in Ks :
                    RA(l)[k] \in ResLo..ResHi /\ RB(l)[k] \in ResLo..ResHi

\* Order-independent: holds for every partial sum in any order.
NoAccOverflow == \A l \in Ls :
                    SumTo([k \in Ks |-> Abs(Prods(l)[k])], K) <= AccMax

CRTExact    == Reconstructed = Exact

=============================================================================
