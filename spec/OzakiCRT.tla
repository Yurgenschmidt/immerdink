----------------------------- MODULE OzakiCRT -----------------------------
(***************************************************************************)
(* Integer core of Ozaki scheme II for ONE output element (DESIGN §3.4,   *)
(* §3.5, §3.6).                                                            *)
(*                                                                         *)
(* After scaling and quantization (DESIGN §3.2-3.3, spec OzakiScaling) the *)
(* row a = A'[i,:] and the column b = B'[:,j] are integer vectors of       *)
(* length K. The scheme computes Exact = sum_k a[k]*b[k] without forming   *)
(* it:                                                                     *)
(*                                                                         *)
(*  §3.4 residues   r_l(x) = SMod(x, p_l) in -(p_l\div 2)..(p_l-1)\div 2   *)
(*                  stored as signed ResBits-bit ints (int8 for real). The *)
(*                  store is modelled as two's-complement truncation       *)
(*                  Op = SMod(r, 2^ResBits) (C++ static_cast<int8_t>): the *)
(*                  identity iff r fits (ResiduesFit), so it only matters  *)
(*                  in the models that drop PreResidue, where it makes the *)
(*                  end-to-end failure the real one (a non-congruent       *)
(*                  operand). A saturating store (cvt.sat) would fail the  *)
(*                  same way: the clamped value is not congruent either.   *)
(*  §3.5 int GEMM   K is cut into chunks of KC consecutive k. Per chunk c  *)
(*                  and modulus l the MMA forms                            *)
(*                     acc = sum_{k in chunk c} Op_l(a[k]) * Op_l(b[k])    *)
(*                  in a signed AccBits-bit accumulator (int32 for real).  *)
(*                  The u8 epilogue stores                                 *)
(*                     c = 1 : W_l <- acc mod p_l                          *)
(*                     c > 1 : W_l <- (W_l + acc) mod p_l                  *)
(*                  (chunks are 1-based here; DESIGN §4's c = 0 / c > 0 is *)
(*                  the same rule 0-based). "mod" is FLOOR mod, result in  *)
(*                  0..p_l-1 for negative arguments too (TLA+ %); a        *)
(*                  truncating rem would give negative values, which a     *)
(*                  saturating u8 store clamps to 0 (TypeOK/WCongruent     *)
(*                  catch it).                                             *)
(*  §3.6 CRT        y_l = (q_l * W_l) mod p_l ; x = (sum_l M_l*y_l) mod P  *)
(*                  z = IF x <= H THEN x ELSE x - P                        *)
(*                                                                         *)
(* HARDWARE ACCUMULATOR MODEL. CUTLASS 2.x int8 mma.sync is .satfinite     *)
(* (DESIGN §2 K_c row) and the MMA accumulation order is unspecified. We   *)
(* do not model either: if the chunk's sum of |products| is <= AccMax then *)
(* acc is the exact sum; otherwise acc is HAVOC, i.e. any value of the     *)
(* accumulator range AccLo..AccMax (this covers saturation, two's-         *)
(* complement wrap, and any order-dependent mixture). The havoc is a       *)
(* nondeterministic choice in the step that processes the chunk, so TLC    *)
(* explores every value.                                                   *)
(*   ORDER-INDEPENDENCE IS AN ENVIRONMENT ASSUMPTION, NOT A CHECKED CLAIM: *)
(* the exact branch is sound for every order because every partial sum of *)
(* any order is bounded in magnitude by sum |products| (no specific order  *)
(* is modelled). What TLC checks is that the design rule PreChunk makes    *)
(* the havoc branch unreachable (ChunkFits), and, in neg_order, that the   *)
(* rule cannot be weakened to "the final chunk sum fits": inputs whose     *)
(* every chunk sum fits but whose sum of |products| does not still break   *)
(* CRTExact (e.g. sequential saturation 16+16+16+16 -> 63, then -12 -> 51  *)
(* instead of 52).                                                         *)
(*                                                                         *)
(* CLAIMS (invariants, see below):                                         *)
(*   ResiduesFit   §3.4  residues fit the signed ResBits operand type      *)
(*   ChunkFits     §3.5  no chunk can overflow (so havoc never fires)      *)
(*   EpilogueFits  §3.5  W + acc never overflows the accumulator type      *)
(*   WCongruent    §3.5  after chunks 1..d, W_l = (partial dot) mod p_l    *)
(*                       (the K-split / chunk-carry claim)                 *)
(*   CRTRecon      §3.6  the §3.6 formula yields THE CRT value: the unique *)
(*                       z in CRange with z = W_l (mod p_l) for all l,     *)
(*                       for EVERY plane tuple, also garbage ones (checked *)
(*                       after reachable havoc in pos_havoc_recon)         *)
(*   CRTReconAbs   §3.6  literally Out = CRTOf(W) (small-P models only)    *)
(*   CRTExact      §3.2+§3.6  SumAbs <= H  =>  z = Exact, checked per      *)
(*                       input while inputs with SumAbs > H are also       *)
(*                       enumerated (the guard is tested on both sides)    *)
(*   CRTExactBeyond      the same with H+1: MUST FAIL (H is tight)         *)
(* plus the constant-level CRTInverse (ASSUME) that both CRT formulas      *)
(* invert the residue map on CRange, which makes it a bijection.          *)
(*                                                                         *)
(* PRECONDITIONS / DESIGN RULES (each has a negative model, see MC):       *)
(*   PreResidue  every p_l <= 2^ResBits         (DESIGN §2: p_l <= 256)    *)
(*               (tight: neg_resbits_p9 has p = 2^ResBits + 1, the analogue *)
(*               of a table starting at 257, whose residue 128 is no int8) *)
(*   PreChunk    KC * 2^(2*(ResBits-1)) <= AccMax (DESIGN §2: K_c =        *)
(*               (2^31-1) \div 128^2 = 131071; §3.5 uses K_c' = 131008)     *)
(*   PreCarry    later chunks ADD into W (ChunkCarry = TRUE)               *)
(*   per input   SumAbs = sum_k |a[k]|*|b[k]| <= H (DESIGN §3.2 guarantee; *)
(*               a guard of CRTExact, not an assumption, so inputs beyond  *)
(*               it are still enumerated).                                 *)
(*                                                                         *)
(* SCALING. TLC integers are 32-bit, so models run at small widths. Every  *)
(* width enters only through the constants (ResBits ~ 8, AccBits ~ 32,     *)
(* KC ~ K_c, Moduli ~ the greedy table of DESIGN §2), and the scaled       *)
(* moduli are produced by the same greedy rule (descending from            *)
(* 2^ResBits, keep a candidate if coprime to all kept ones):               *)
(*   ResBits = 2 -> <<4,3>>   ResBits = 3 -> <<8,7,5,3>>                    *)
(*   ResBits = 4 -> <<16,15,13,11,7>>   (models use prefixes, like N)       *)
(* so each model has the 256 analogue p = 2^ResBits (asymmetric residue    *)
(* range, even P) and odd moduli. The real-width instances of the same     *)
(* closed-form conditions are certified by spec/params.py.                 *)
(*                                                                         *)
(* Step 4 is EXACT integer arithmetic here. Its multi-word running-mod     *)
(* realisation (L_w-bit words) is spec CRTRealization.                     *)
(***************************************************************************)
EXTENDS Integers, Sequences

CONSTANTS
    Moduli,      \* <<p_1, ..., p_N>>, pairwise coprime (a greedy-table prefix)
    ResBits,     \* signed bits of a residue operand          (real: 8)
    AccBits,     \* signed bits of the MMA accumulator        (real: 32)
    K,           \* contraction length
    KC,          \* K-chunk length                            (real: K_c')
    ValsA,       \* values enumerated for each a[k] (finite set of Int)
    ValsB,       \* values enumerated for each b[k]
    ChunkCarry,  \* TRUE: design epilogue; FALSE: later chunks overwrite W
    Drop,        \* "none" | "PreResidue" | "PreChunk" | "PreCarry":
                 \*   the ONE precondition a negative model violates
    InputFilter, \* "all" | "finalSumFits": restrict the enumerated inputs
                 \*   (negative models only, see the ASSUME below)
    Covers       \* subset of {"H", "KC", "carry", "havoc"}: what this model
                 \*   is required (by ASSUME) to reach, see Coverage

VARIABLES
    a, b,        \* the input, chosen in Init
    step,        \* number of (chunk, modulus) GEMM steps done
    W            \* W[l]: the u8 plane value of modulus l (Unwritten at start)

vars == <<a, b, step, W>>

-----------------------------------------------------------------------------
(* Arithmetic helpers *)

RECURSIVE ProdSeq(_)
ProdSeq(s) == IF s = <<>> THEN 1 ELSE Head(s) * ProdSeq(Tail(s))

\* Sum of f[i] for i in lo..hi (f a function whose domain contains lo..hi).
RECURSIVE SumF(_, _, _)
SumF(f, lo, hi) == IF lo > hi THEN 0 ELSE f[lo] + SumF(f, lo + 1, hi)

Abs(x)    == IF x < 0 THEN -x ELSE x
Min(x, y) == IF x < y THEN x ELSE y

\* Signed residue (DESIGN §3.4): SMod(x, m) in -(m \div 2) .. (m - 1) \div 2.
\* m = 256 -> -128..127 (exactly int8); m odd -> symmetric.
SMod(x, m) == LET r == x % m IN IF r >= (m + 1) \div 2 THEN r - m ELSE r

Coprime(x, y) == \A d \in 2..Min(x, y) : ~(x % d = 0 /\ y % d = 0)
Inv(x, m)     == CHOOSE y \in 1..(m - 1) : (x * y) % m = 1

-----------------------------------------------------------------------------
(* Derived constants. All zero-arity, so TLC evaluates each once.         *)

N      == Len(Moduli)
Ls     == 1..N
Ks     == 1..K
P      == ProdSeq(Moduli)                     \* DESIGN §2 P
H      == (P - 1) \div 2                      \* DESIGN §2 H
Mh     == [l \in Ls |-> P \div Moduli[l]]     \* M_l
Q      == [l \in Ls |-> Inv(Mh[l] % Moduli[l], Moduli[l])]  \* q_l in 1..p_l-1
Wt     == [l \in Ls |-> Mh[l] * Q[l]]         \* classical CRT weight M_l*q_l

\* The centred CRT range of DESIGN §3.6: z = x if x <= H, else x - P, with
\* x in 0..P-1, i.e. z in (H+1-P)..H. For even P (always: 2^ResBits is in
\* every table prefix) that is -(H+1)..H = -P/2 .. P/2-1, the same
\* "extra value on the negative side" convention as SMod.
CRange == (H + 1 - P)..H

ResMag == 2^(ResBits - 1)                     \* max |residue|, incl. p = 2^ResBits
ResLo  == -ResMag
ResHi  == ResMag - 1
AccMax == 2^(AccBits - 1) - 1
AccLo  == -AccMax - 1
KCMax  == AccMax \div (ResMag * ResMag)       \* real: 131071

C      == (K + KC - 1) \div KC                \* number of chunks
Cs     == 1..C
ChunkLo(c) == (c - 1) * KC + 1
ChunkHi(c) == Min(c * KC, K)
ChunkOf(k) == ((k - 1) \div KC) + 1

NSteps == N * C
\* Steps run chunk-major (for c: for l), as K4 does within a residue group.
\* Step s (0-based) processes modulus StepL(s) on chunk StepC(s).
StepL(s) == (s % N) + 1
StepC(s) == (s \div N) + 1
\* Number of chunks of modulus l already folded into W[l] after `st` steps.
DoneChunks(st, l) == (st + N - l) \div N

Unwritten == -1

-----------------------------------------------------------------------------
(* The CRT, abstractly and as the two concrete formulas.                  *)

\* Residue map: the u8 values W_l that an integer z produces.
Enc(z) == [l \in Ls |-> z % Moduli[l]]

\* ABSTRACT CRT (statement only; never evaluated at scale): the unique z
\* in CRange congruent to w[l] mod p_l for every l. Uniqueness/existence on
\* CRange is established by CRTInverse below.
CRTOf(w) == CHOOSE z \in CRange : Enc(z) = w

Center(x) == IF x <= H THEN x ELSE x - P

\* DESIGN §3.6 as written: pre-reduce y_l = (q_l * W_l) mod p_l.
ReconD(w) == Center(SumF([l \in Ls |-> Mh[l] * ((Q[l] * w[l]) % Moduli[l])], 1, N) % P)
\* Classical weights formula: sum_l (M_l q_l) W_l mod P.
ReconW(w) == Center(SumF([l \in Ls |-> Wt[l] * w[l]], 1, N) % P)

-----------------------------------------------------------------------------
(* Input functionals (pure functions of a, b).                            *)

R(l, x)       == SMod(x, Moduli[l])
\* The stored ResBits-bit operand: two's-complement truncation of R(l, x).
\* Equal to R(l, x) exactly when R(l, x) \in ResLo..ResHi (ResiduesFit).
Op(l, x)      == SMod(R(l, x), 2 * ResMag)
Prod(l, k)    == Op(l, a[k]) * Op(l, b[k])
ProdF(l)      == [k \in Ks |-> Prod(l, k)]
AbsProdF(l)   == [k \in Ks |-> Abs(Prod(l, k))]
ChunkSum(l, c) == SumF(ProdF(l), ChunkLo(c), ChunkHi(c))
ChunkAbs(l, c) == SumF(AbsProdF(l), ChunkLo(c), ChunkHi(c))

DotF    == [k \in Ks |-> a[k] * b[k]]
Exact   == SumF(DotF, 1, K)
SumAbs  == SumF([k \in Ks |-> Abs(a[k]) * Abs(b[k])], 1, K)
\* Exact dot product over chunks 1..d.
Partial(d) == SumF(DotF, 1, ChunkHi(d))

-----------------------------------------------------------------------------
(* Assumptions                                                             *)

WellFormed ==
    /\ N >= 1 /\ K >= 1 /\ KC >= 1 /\ ResBits >= 1 /\ AccBits >= 2
    /\ \A l \in Ls : Moduli[l] >= 2
    /\ \A i, j \in Ls : i # j => Coprime(Moduli[i], Moduli[j])
    /\ ValsA # {} /\ ValsB # {}
    /\ ChunkCarry \in BOOLEAN
    /\ Drop \in {"none", "PreResidue", "PreChunk", "PreCarry"}
    /\ InputFilter \in {"all", "finalSumFits"}
    /\ Covers \subseteq {"H", "KC", "carry", "havoc"}

\* DESIGN §2/§3.4: residues fit the signed ResBits-bit operand type.
PreResidue == \A l \in Ls : Moduli[l] <= 2^ResBits
\* DESIGN §2/§3.5: KC * max|r|^2 <= AccMax, with max|r| = 2^(ResBits-1)
\* (attained only by p = 2^ResBits, whose SMod range is asymmetric).
PreChunk   == KC * ResMag * ResMag <= AccMax
\* DESIGN §3.5: later chunks add into W instead of overwriting it.
PreCarry   == ChunkCarry

ASSUME WellFormed
\* Every precondition holds, except exactly the one a negative model drops
\* (and that one really fails, so each negative model isolates one rule).
ASSUME PreResidue = (Drop # "PreResidue")
ASSUME PreChunk   = (Drop # "PreChunk")
ASSUME PreCarry   = (Drop # "PreCarry")
\* An input filter only ever narrows a NEGATIVE model (one that drops a
\* rule), so it cannot weaken any positive claim.
ASSUME InputFilter # "all" => Drop # "none"

\* CRT, constant level (checked once per model over all of CRange): both
\* formulas are left inverses of Enc on CRange. Hence Enc is injective on
\* CRange; |CRange| = P = |product of 0..p_l-1|, so Enc is a bijection and
\* ReconD = ReconW = CRTOf on EVERY residue tuple. Real width: the δ_lj
\* certificate in spec/params.py.
CRTInverse == \A z \in CRange : ReconD(Enc(z)) = z /\ ReconW(Enc(z)) = z
ASSUME CRTInverse

\* Coverage: what a model claims to exercise, so "positive" really sits on
\* the boundary rather than comfortably inside it.
AbsProds == {Abs(x) * Abs(y) : x \in ValsA, y \in ValsB}
RECURSIVE SumsOf(_)
SumsOf(n) == IF n = 0 THEN {0} ELSE {s + t : s \in SumsOf(n - 1), t \in AbsProds}
ReachSumAbs == SumsOf(K)
Coverage ==
    \* inputs with SumAbs = H and with SumAbs = H + 1 are enumerated
    /\ "H" \in Covers => {H, H + 1} \subseteq ReachSumAbs
    \* KC is the largest legal chunk and a chunk of KC worst-case products
    \* (|r_a| = |r_b| = 2^(ResBits-1)) is enumerated
    /\ "KC" \in Covers =>
          /\ KC = KCMax /\ K >= KC
          /\ \E l \in Ls, x \in ValsA, y \in ValsB :
                Abs(R(l, x) * R(l, y)) = ResMag * ResMag
    \* the chunk carry is exercised
    /\ "carry" \in Covers => C >= 2
    \* the havoc branch is reachable: a full chunk of KC worst-case
    \* products (sum KC * ResMag^2 > AccMax) is enumerated
    /\ "havoc" \in Covers =>
          /\ ~PreChunk /\ K >= KC
          /\ \E l \in Ls, x \in ValsA, y \in ValsB :
                Abs(Op(l, x) * Op(l, y)) = ResMag * ResMag
ASSUME Coverage

-----------------------------------------------------------------------------
(* Behaviour                                                               *)

LexLeq(x, y) == x[1] < y[1] \/ (x[1] = y[1] /\ x[2] <= y[2])

\* Symmetry reduction: all claims are invariant under permuting the k's
\* WITHIN one chunk (acc is a sum; the havoc condition is a sum of
\* |products|), so only inputs whose (a[k], b[k]) pairs are sorted inside
\* each chunk are enumerated. Pairs in different chunks are not reordered.
Canonical == \A k \in 1..(K - 1) :
                ChunkOf(k) = ChunkOf(k + 1) =>
                    LexLeq(<<a[k], b[k]>>, <<a[k + 1], b[k + 1]>>)

\* "finalSumFits": only inputs for which every chunk's FINAL sum fits the
\* accumulator for every modulus (so a hardware model that only needed the
\* final sum to fit would compute them exactly). Used by neg_order.
FilterOK ==
    InputFilter = "finalSumFits" =>
        \A l \in Ls, c \in Cs : Abs(ChunkSum(l, c)) <= AccMax

Init ==
    /\ a \in [Ks -> ValsA]
    /\ b \in [Ks -> ValsB]
    /\ Canonical
    /\ FilterOK
    /\ step = 0
    /\ W = [l \in Ls |-> Unwritten]

\* Hardware accumulator for chunk c of modulus l: exact if no partial sum
\* can leave the range, otherwise any value of the range (havoc).
AccChoices(l, c) ==
    IF ChunkAbs(l, c) <= AccMax THEN {ChunkSum(l, c)} ELSE AccLo..AccMax

\* u8 epilogue (DESIGN §3.5). % is floor-mod: result in 0..p-1 also for
\* negative arguments, as the epilogue's mod-reduction must be.
Epi(l, c, acc) ==
    IF c = 1 \/ ~ChunkCarry THEN acc % Moduli[l]
                            ELSE (W[l] + acc) % Moduli[l]

GemmStep ==
    /\ step < NSteps
    /\ LET l == StepL(step)
           c == StepC(step)
       IN \E acc \in AccChoices(l, c) :
             W' = [W EXCEPT ![l] = Epi(l, c, acc)]
    /\ step' = step + 1
    /\ UNCHANGED <<a, b>>

Next == GemmStep
Spec == Init /\ [][Next]_vars

Done == step = NSteps
Out  == ReconD(W)                              \* the §3.6 result z

-----------------------------------------------------------------------------
(* Invariants. Input-only claims are checked in the step-0 state, which   *)
(* exists exactly once per input.                                         *)

TypeOK ==
    /\ a \in [Ks -> ValsA] /\ b \in [Ks -> ValsB]
    /\ step \in 0..NSteps
    /\ \A l \in Ls : W[l] = Unwritten \/ W[l] \in 0..(Moduli[l] - 1)

\* §3.4: every residue fits the signed ResBits-bit operand (int8 for real),
\* i.e. the operand store Op is the identity on every residue.
ResiduesFit ==
    step = 0 =>
        \A l \in Ls, k \in Ks :
            R(l, a[k]) \in ResLo..ResHi /\ R(l, b[k]) \in ResLo..ResHi

\* §3.5: sum of |products| of every chunk <= AccMax, so every partial sum
\* in every accumulation order is representable and .satfinite never
\* triggers (equivalently: the havoc branch is unreachable).
ChunkFits ==
    step = 0 => \A l \in Ls, c \in Cs : ChunkAbs(l, c) <= AccMax

\* §3.5: the continuation add W + acc (W in 0..p-1) stays in the
\* accumulator type for every chunk after the first. (Implied by PreChunk
\* when AccBits-1 >= 2*(ResBits-1) and ResBits >= 2; real width: [cert].)
EpilogueFits ==
    step = 0 =>
        \A l \in Ls, c \in 2..C : (Moduli[l] - 1) + ChunkAbs(l, c) <= AccMax

\* §3.5 K-split: after d chunks of modulus l, W[l] is the partial exact
\* dot product over those chunks, mod p_l (and Unwritten before chunk 1).
WCongruent ==
    \A l \in Ls :
        LET d == DoneChunks(step, l)
        IN IF d = 0 THEN W[l] = Unwritten
                    ELSE W[l] = Partial(d) % Moduli[l]

\* §3.6: the reconstruction is the abstract CRT value of the planes:
\* in CRange and congruent to every W_l (unique by CRTInverse). Follows
\* from CRTInverse for every tuple; checked in reachable final states,
\* including after havoc (pos_havoc_recon), where W is garbage.
CRTRecon ==
    Done => Out \in CRange /\ Enc(Out) = W

\* §3.6, literal form for small P only (O(P) CHOOSE per final state): the
\* §3.6 formula equals the abstract CRT value CRTOf(W).
CRTReconAbs ==
    Done => Out = CRTOf(W)

\* §3.2 + §3.6 main claim: per input, SumAbs <= H implies exactness.
CRTExact ==
    Done /\ SumAbs <= H => Out = Exact

\* NEGATIVE ONLY: the bound cannot be relaxed to H + 1 (P is even, so
\* Exact = H + 1 = P/2 reconstructs to -P/2).
CRTExactBeyond ==
    Done /\ SumAbs <= H + 1 => Out = Exact

=============================================================================
