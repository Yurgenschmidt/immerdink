--------------------------- MODULE CRTRealization ---------------------------
(***************************************************************************)
(* DESIGN.md §3.6: division-free running CRT in a bounded unsigned word,  *)
(* realized limb by limb, and its REFINEMENT of the exact (abstract) CRT. *)
(*                                                                         *)
(* Maps to DESIGN.md:                                                     *)
(*   §2   p_l (Moduli), P, H = (P-1) \div 2, M_l = P/p_l, q_l = M_l^-1    *)
(*        mod p_l, L_w (WordBits) with the rule 2P <= 2^L_w.              *)
(*   §3.2 the contract |A'B'| <= H that CRT must reconstruct exactly.     *)
(*   §3.5 the inputs: u8 residue planes W_l \in [0, p_l) (variable w).    *)
(*   §3.6 the algorithm, verbatim:                                         *)
(*          x <- 0                                                         *)
(*          for l = 1..N:  y <- (q_l * W_l) mod p_l      (8-bit x 8-bit)   *)
(*                         x <- x + M_l * y              (L_w-bit word)    *)
(*                         if x >= P: x <- x - P                           *)
(*          z <- (x <= H) ? x : x - P                    (centered)        *)
(*        "exact, multi-word": the L_w-bit word is a vector of limbs.      *)
(*                                                                         *)
(* Three layers, each checked against the one above it by TLC:             *)
(*                                                                         *)
(*  CRTExact (inner module, abstract). From 'ready', ONE step to 'done'   *)
(*    that produces the unique centered integer                            *)
(*        z \in -(P-1-H) .. H   with   z = W_l (mod p_l) for every l.      *)
(*    It is defined by congruence + uniqueness only. It never mentions    *)
(*    M_l, q_l or any CRT weight, so the refinement is a real check of the *)
(*    weight formula. Uniqueness (CRT for these moduli) and the §3.2       *)
(*    contract (every |c| <= H is representable) are checked by TLC as     *)
(*    ASSUMEs re-asserted below.                                           *)
(*                                                                         *)
(*  CRTWord (inner module). The §3.6 pseudocode with x ONE WordBits-bit    *)
(*    unsigned integer: Word(v) wraps mod 2^WordBits as unsigned           *)
(*    __int128 / a 160-bit unsigned type would, and the ghost ovf records  *)
(*    that an exact value did not fit. With a too-narrow word the machine  *)
(*    keeps running on wrapped values, as hardware would.                  *)
(*                                                                         *)
(*  This module (concrete). The multi-word realization: x is a tuple of    *)
(*    Limbs limbs (little-endian, LimbBits each; the top limb holds the    *)
(*    remaining WordBits - (Limbs-1)*LimbBits bits). Each §3.6 line is a   *)
(*    sequence of limb micro-steps, as device code would execute it:       *)
(*      mac  x <- x + M_l*y   one fused carry chain, limb k:               *)
(*                            t = x_k + M_l,k * y + c; x_k = t mod 2^b_k;  *)
(*                            c = t div 2^b_k; the top carry-out is lost   *)
(*                            (unsigned wrap) and sets ovf.                *)
(*      cmp  x >= P / x > H   lexicographic, from the top limb down.       *)
(*      sub  x <- x - P       one borrow chain; the top borrow-out is lost *)
(*                            (two's-complement wrap when centering).      *)
(*      center: the sign of z is the top bit of the top limb.             *)
(*                                                                         *)
(*  Refinement mappings (checked as PROPERTIES):                           *)
(*    RefinesWord: limbs => CRTWord. XWord recovers the word value of x    *)
(*      in the middle of a carry/borrow chain from the chain identity      *)
(*        mac at limb k:  x_old = Value(x) + c*2^(b(k-1)) - (M_l mod       *)
(*                                 2^(b(k-1))) * y                         *)
(*        sub at limb k:  x_old = Value(x) + (P mod 2^(b(k-1)))            *)
(*                                 - c*2^(b(k-1)),                         *)
(*      so every micro-step but the last of a chain must be a word-level  *)
(*      stutter. A dropped carry, a dropped borrow or a wrong compare     *)
(*      changes XWord on a stutter step and is caught (neg_mac_nocarry,   *)
(*      neg_sub_noborrow, neg_cmp_toponly).                               *)
(*    Refines: limbs => CRTExact (z read off the word at 'done').          *)
(*  Every behavior of CRTWord is the image of a limb behavior (both are   *)
(*  deterministic and every input is an initial state), so the pair also  *)
(*  checks CRTWord => CRTExact.                                            *)
(*                                                                         *)
(* Scaling. TLC integers are 32-bit, so the models run at scaled widths,  *)
(* with moduli built by the SAME greedy rule as DESIGN §2 (pairwise-      *)
(* coprime, descending from 2^ResBits):                                    *)
(*   ResBits = 8 (real): 256, 255, ...  L_w = 128 / 160 = 4 / 5 limbs x 32 *)
(*   ResBits = 3: <<8, 7, 5, 3>>      P =   840, L_w = 11 = 4+4+3 or 6+5   *)
(*   ResBits = 4: <<16, 15, 13, 11>>  P = 34320, L_w = 17 = 6+6+5          *)
(* In both scaled sets p_1 = 2^ResBits is even, as 256 is for real. The   *)
(* scaled minimal L_w (11, 17) are prime, so the scaled top limb is       *)
(* narrower than the others; real L_w (128, 160) are whole 32-bit limbs,  *)
(* the special case top limb = LimbBits of the same micro-steps;          *)
(* pos_8753_l1 is the one-limb (native word) case. WordBits in the        *)
(* positive models is the least width                                      *)
(* allowed by 2P <= 2^WordBits; neg_word_* use one bit less. Every       *)
(* residue vector (prod p_l of them) is an initial state, so each claim is *)
(* checked for ALL inputs of the scaled instance.                          *)
(*                                                                         *)
(* PeakExact = max_l (2P - prod_{j>=l} p_j - M_l) is the exact largest    *)
(* intermediate (x + M_l*y) over all inputs: PeakBound shows it is an     *)
(* upper bound, and the neg_peak_attained models show it is attained.     *)
(* spec/params.py evaluates the same formula at real width ([cert]).      *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

    ----------------------------- MODULE CRTExact -----------------------------
    (***********************************************************************)
    (* Abstract exact CRT (DESIGN §3.6 "z = (A'B') exactly", §2 H).       *)
    (* w (the residue planes of §3.5) is a VARIABLE, kept UNCHANGED: a     *)
    (* CONSTANT could not be mapped to the concrete state.                 *)
    (***********************************************************************)
    EXTENDS Integers, Sequences, FiniteSets
    CONSTANT Moduli
    VARIABLES w, phase, z
    absVars == <<w, phase, z>>

    RECURSIVE ProdSeq(_)
    ProdSeq(s) == IF s = <<>> THEN 1 ELSE Head(s) * ProdSeq(Tail(s))

    N       == Len(Moduli)
    Ls      == 1..N
    P       == ProdSeq(Moduli)
    H       == (P - 1) \div 2                    \* DESIGN §2
    ZRange  == (-(P - 1 - H))..H                 \* centered range, |ZRange| = P
    MaxMod  == CHOOSE m \in {Moduli[l] : l \in Ls} : \A l \in Ls : Moduli[l] <= m
    ResVecs == {v \in [Ls -> 0..(MaxMod - 1)] : \A l \in Ls : v[l] < Moduli[l]}

    \* Cheap membership test (ResVecs is enumerated only for Unique/Inputs).
    IsResVec(v) == v \in [Ls -> Int] /\ \A l \in Ls : 0 <= v[l] /\ v[l] < Moduli[l]
    ResOf(c)    == [l \in Ls |-> c % Moduli[l]]
    \* The congruence the abstract step uses IS the map that Unique certifies.
    IsCRT(c, v) == ResOf(c) = v

    (* CRT existence + uniqueness for THESE moduli, checked by enumeration:*)
    (* c |-> ResOf(c) maps the P-element ZRange into the P-element         *)
    (* ResVecs, and its image has P elements, so it is a bijection. Hence  *)
    (* every residue vector has exactly one centered preimage.             *)
    Unique == /\ Cardinality(ZRange) = P
              /\ Cardinality(ResVecs) = P
              /\ Cardinality({ResOf(c) : c \in ZRange}) = P
    (* DESIGN §2/§3.2 contract: scaling guarantees |A'B'| <= H, so every   *)
    (* such value must lie in the range the reconstruction produces.       *)
    Contract == (-H)..H \subseteq ZRange
    ASSUME Unique /\ Contract  \* NB: TLC skips this when INSTANCEd; re-asserted below

    Init == /\ IsResVec(w)
            /\ phase = "ready"
            /\ z = 0
    \* Relational form of z' = the unique preimage: equivalent under Unique,
    \* and it costs TLC O(N) per transition instead of a CHOOSE over ZRange.
    Reconstruct == /\ phase = "ready"
                   /\ phase' = "done"
                   /\ z' \in ZRange
                   /\ IsCRT(z', w)
                   /\ UNCHANGED w
    Spec       == Init /\ [][Reconstruct]_absVars /\ WF_absVars(Reconstruct)
    SafetySpec == Init /\ [][Reconstruct]_absVars
    ===========================================================================

    ------------------------------ MODULE CRTWord ------------------------------
    (***********************************************************************)
    (* DESIGN §3.6 at word level: x is one WordBits-bit unsigned integer.  *)
    (* The §2 tables are CONSTANT parameters, supplied by the outer module *)
    (* (where TLC computes them once).                                     *)
    (***********************************************************************)
    EXTENDS Integers
    CONSTANTS Moduli, N, P, H, Mh, Q, ResBits, WordBits,
              UseQ, UseCondSub, CenterInclusive
    VARIABLES w, l, stage, y, x, ovf
    wvars == <<w, l, stage, y, x, ovf>>

    WordMod  == 2^WordBits
    Word(v)  == v % WordMod                                  \* unsigned wrap
    Fits(v)  == 0 <= v /\ v < WordMod
    SFits(v) == -(2^(WordBits - 1)) <= v /\ v < 2^(WordBits - 1)

    Init == /\ DOMAIN w = 1..N /\ \A i \in 1..N : w[i] \in 0..(Moduli[i] - 1)
            /\ l = 1 /\ stage = "mul" /\ y = 0 /\ x = 0 /\ ovf = FALSE

    \* y <- (q_l * W_l) mod p_l   (an 8-bit x 8-bit product)
    Mul == /\ stage = "mul"
           /\ y' = IF UseQ THEN (Q[l] * w[l]) % Moduli[l] ELSE w[l]
           /\ stage' = "add"
           /\ UNCHANGED <<w, l, x, ovf>>
    \* x <- x + M_l * y   in the word
    Add == /\ stage = "add"
           /\ LET s == x + Mh[l] * y IN
                /\ x' = Word(s)
                /\ ovf' = (ovf \/ ~Fits(Mh[l] * y) \/ ~Fits(s))
           /\ stage' = "red"
           /\ UNCHANGED <<w, l, y>>
    \* if x >= P: x <- x - P   (never negative, so no wrap here)
    Red == /\ stage = "red"
           /\ x' = IF UseCondSub /\ x >= P THEN x - P ELSE x
           /\ l' = l + 1
           /\ stage' = IF l = N THEN "center" ELSE "mul"
           /\ UNCHANGED <<w, y, ovf>>
    \* z <- (x <= H) ? x : x - P, stored as two's complement in the same word
    Center == /\ stage = "center"
              /\ LET neg == IF CenterInclusive THEN x > H ELSE x >= H
                     zv  == IF neg THEN x - P ELSE x
                 IN /\ x' = Word(zv)
                    /\ ovf' = (ovf \/ ~SFits(zv))
              /\ stage' = "done"
              /\ UNCHANGED <<w, l, y>>
    Next == Mul \/ Add \/ Red \/ Center
    Spec       == Init /\ [][Next]_wvars /\ WF_wvars(Next)
    SafetySpec == Init /\ [][Next]_wvars

    \* Word-level invariants (checked through the mapping, see below).
    TypeOK == /\ l \in 1..(N + 1)
              /\ stage \in {"mul", "add", "red", "center", "done"}
              /\ x \in 0..(WordMod - 1)
              /\ ovf \in BOOLEAN
    \* §3.6 "(8-bit x 8-bit)": both factors of q_l * W_l, and y, fit ResBits
    \* unsigned bits, so the product fits 2*ResBits (16 bits real).
    ByteOperands == stage \in {"mul", "add"} =>
                        /\ w[l] < 2^ResBits /\ Q[l] < 2^ResBits /\ y < 2^ResBits
    \* §3.6 "M_l*y < P": the addend, exactly; indeed M_l*y <= P - M_l.
    AddendBound == stage = "red" => Mh[l] * y <= P - Mh[l]
    \* §3.6 "invariant x < P": holds at every loop head (after each full step).
    XBelowP == stage \in {"mul", "center"} => x < P
    \* The inductive core of the algorithm, and the reason q_l is needed:
    \* after l-1 steps x is congruent to W_i mod p_i for every finished i and
    \* to 0 for every pending i (M_l = 0 mod p_i for i # l, q_l*M_l = 1 mod p_l).
    RunningCRT == stage \in {"mul", "center"} =>
                    \A i \in 1..N : x % Moduli[i] = IF i < l THEN w[i] ELSE 0
    \* §3.6 "peak < 2P <= 2^L_w": no exact intermediate x + M_l*y leaves the
    \* word. (The signed-range arm SFits of Center cannot fire under the L_w
    \* rule, since |z| <= P/2 <= 2^(L_w-2); it is kept as the definition of
    \* a correct two's-complement read, not as a separately tested claim.)
    NoWordOverflow == ~ovf
    ===========================================================================

CONSTANTS
    Moduli,               \* <<p_1, ..., p_N>>, DESIGN §2 (scaled)
    ResBits,              \* bits of the residue/q operands: 8 real ("8-bit x 8-bit")
    WordBits,             \* L_w: CRT word width; 128 (N <= 16) / 160 (N <= 20) real
    LimbBits,             \* limb width: 32 real (L_w = 4 / 5 limbs)
    UseQ,                 \* design rule: y = (q_l * W_l) mod p_l   (FALSE: y = W_l)
    UseCondSub,           \* design rule: if x >= P then x <- x - P (FALSE: dropped)
    CenterInclusive,      \* design rule: z = (x <= H) ? x : x-P    (FALSE: x < H)
    Broken,               \* which DESIGN precondition this model violates:
                          \*   "none" | "coprime" | "resbits" | "word"
    LimbBug               \* limb-realization mutation (negative models only):
                          \*   "none" | "mac_nocarry" | "sub_noborrow" | "cmp_toponly"

VARIABLES
    w,       \* residue planes W_l \in [0, p_l)  (input, frozen)
    l,       \* loop index, 1..N+1 (N+1: centering)
    stage,   \* "mul" | "mac" | "cmp" | "sub" | "done"
    y,       \* (q_l * W_l) mod p_l
    x,       \* the CRT word: Limbs limbs, little-endian (holds z's bits at "done")
    k,       \* limb index of the running chain (mac/sub: 1 up; cmp: Limbs down)
    c,       \* carry (mac) / borrow (sub) into limb k
    ovf,     \* ghost: a carry/borrow left the word, or the sign is wrong
    peak     \* ghost: largest exact intermediate x + M_l*y seen so far
vars == <<w, l, stage, y, x, k, c, ovf, peak>>

-----------------------------------------------------------------------------
(* Constants of DESIGN §2 (zero-arity, so TLC computes each table once) *)

AbsC == INSTANCE CRTExact WITH Moduli <- Moduli, w <- w, phase <- "ready", z <- 0

N       == AbsC!N
Ls      == AbsC!Ls
P       == AbsC!P
H       == AbsC!H
ZRange  == AbsC!ZRange
MaxMod  == AbsC!MaxMod
Mh      == [i \in Ls |-> P \div Moduli[i]]                       \* M_l
Q       == [i \in Ls |->                                          \* q_l = M_l^-1 mod p_l
              IF \E q \in 1..(Moduli[i] - 1) : (Mh[i] * q) % Moduli[i] = 1
              THEN CHOOSE q \in 1..(Moduli[i] - 1) : (Mh[i] * q) % Moduli[i] = 1
              ELSE 0]                                             \* (only if not coprime)
Max(a, b) == IF a > b THEN a ELSE b
Coprime(a, b) == \A d \in 2..a : ~(a % d = 0 /\ b % d = 0)

\* The inputs: every residue vector (enumerated once; TLC caches it).
Inputs  == AbsC!ResVecs

\* prod_{j >= i} p_j. Before step i, x is a multiple of it (RunningCRT), so
\* x <= P - Tail(i); and M_i*y <= P - M_i. Both maxima are reachable at once.
TailProd(i) == AbsC!ProdSeq(SubSeq(Moduli, i, N))
PeakExact   == LET pk == [i \in Ls |-> 2 * P - TailProd(i) - Mh[i]]
               IN CHOOSE v \in {pk[i] : i \in Ls} : \A i \in Ls : pk[i] <= v

(* Limb layout: Limbs limbs, all LimbBits wide except the top one. *)
Limbs   == (WordBits + LimbBits - 1) \div LimbBits
LK      == 1..Limbs
LBits   == [j \in LK |-> IF j < Limbs THEN LimbBits ELSE WordBits - (Limbs - 1) * LimbBits]
Wt      == [j \in LK |-> 2^(LimbBits * (j - 1))]     \* weight of limb j = 2^(b(j-1))
ToLimbs(v) == [j \in LK |-> (v \div Wt[j]) % 2^LBits[j]]
RECURSIVE ValueTo(_, _)
ValueTo(xs, j) == IF j = 0 THEN 0 ELSE xs[j] * Wt[j] + ValueTo(xs, j - 1)
Value(xs) == ValueTo(xs, Limbs)
SignBit(xs) == xs[Limbs] \div 2^(LBits[Limbs] - 1)  \* bit L_w - 1 of the word
MhL     == [i \in Ls |-> ToLimbs(Mh[i])]             \* M_l in limbs
PL      == ToLimbs(P)
HL      == ToLimbs(H)

WordMod  == 2^WordBits
Signed(u) == IF u >= 2^(WordBits - 1) THEN u - WordMod ELSE u   \* two's complement read

-----------------------------------------------------------------------------
(* Assumptions. Each is an Assert with its own message, so a negative model *)
(* can EXPECT exactly the assumption it is about.                           *)

WellFormed ==
    /\ N >= 1
    /\ \A i \in Ls : Moduli[i] >= 2
    /\ ResBits >= 1
    /\ WordBits \in 2..24          \* keeps every TLC intermediate inside int32
    /\ LimbBits \in 1..WordBits /\ LimbBits + ResBits <= 24
    /\ P < 2^WordBits              \* P, H, M_l are representable in the word
    /\ UseQ \in BOOLEAN /\ UseCondSub \in BOOLEAN /\ CenterInclusive \in BOOLEAN
    /\ Broken \in {"none", "coprime", "resbits", "word"}
    /\ LimbBug \in {"none", "mac_nocarry", "sub_noborrow", "cmp_toponly"}

\* The DESIGN preconditions, one predicate per rule.
CoprimeRule == \A i, j \in Ls : i # j => Coprime(Moduli[i], Moduli[j])  \* DESIGN §2
ResBitsRule == \A i \in Ls : Moduli[i] <= 2^ResBits                     \* W_l, q_l are u8
WordRule    == 2 * P <= 2^WordBits                                      \* DESIGN §2 L_w rule

ASSUME Assert(WellFormed, "WellFormed is false")
\* Each rule holds exactly when the model does not claim to break it. Positive
\* models (Broken = "none") need all three; a negative model must violate
\* exactly its own rule, so a weakened rule (e.g. P <= 2^L_w, or a Coprime
\* that misses a factor) makes that negative fail HERE instead of with its
\* expected counterexample.
ASSUME Assert(CoprimeRule = (Broken # "coprime"), "CoprimeRule does not match Broken")
ASSUME Assert(ResBitsRule = (Broken # "resbits"), "ResBitsRule does not match Broken")
ASSUME Assert(WordRule    = (Broken # "word"),    "WordRule does not match Broken")
\* The limb layout really is the WordBits-bit word (top limb 1..LimbBits bits).
ASSUME Assert(LBits[Limbs] \in 1..LimbBits /\ Value(ToLimbs(P)) = P,
              "limb layout does not represent the word")
\* §2/§3.2: H is the DESIGN value (stated independently of CRTExact), and
\* every |c| <= H is in the centered range.
ASSUME Assert(H = (P - 1) \div 2 /\ AbsC!Contract, "AbsContract: -H..H not in ZRange")
\* Init enumerates ALL inputs: Inputs are residue vectors and there are
\* P = prod p_l of them, i.e. all of them.
ASSUME Assert((\A v \in Inputs : AbsC!IsResVec(v)) /\ Cardinality(Inputs) = P,
              "Inputs does not enumerate every residue vector")
\* Lemma: the exact peak is below 2P, so the L_w rule 2P <= 2^L_w suffices.
ASSUME Assert(PeakExact < 2 * P, "PeakExact >= 2P")
\* Abstract well-definedness (TLC does not evaluate ASSUMEs of INSTANCEd
\* modules, so they are re-asserted here). Unique fails iff the moduli are not
\* pairwise coprime: see neg_coprime. It is the LAST assumption, so when it
\* is the one reported, every assumption above it held.
ASSUME Assert(AbsC!Unique, "AbsUnique: CRT preimage not unique")

-----------------------------------------------------------------------------
(* The §3.6 state machine, limb by limb *)

Init == /\ w \in Inputs           \* every residue vector: prod p_l states
        /\ l = 1
        /\ stage = "mul"
        /\ y = 0
        /\ x = [j \in LK |-> 0]
        /\ k = 1
        /\ c = 0
        /\ ovf = FALSE
        /\ peak = 0

(* Refinement mapping to CRTWord: the word value of x, also in the middle   *)
(* of a carry/borrow chain (the chain identity; see the header).            *)
XWord == CASE stage = "mac" -> Value(x) + c * Wt[k] - (Mh[l] % Wt[k]) * y
           [] stage = "sub" -> Value(x) + (P % Wt[k]) - c * Wt[k]
           [] OTHER         -> Value(x)
StageW == CASE stage = "mul"                     -> "mul"
            [] stage = "mac"                     -> "add"
            [] stage \in {"cmp", "sub"} /\ l <= N -> "red"
            [] stage \in {"cmp", "sub"}          -> "center"
            [] stage = "done"                    -> "done"

\* After step l: the next loop head (or the centering compare).
NextHead == /\ l' = l + 1
            /\ stage' = IF l = N THEN "cmp" ELSE "mul"
            /\ k' = IF l = N THEN Limbs ELSE 1
            /\ c' = 0

\* y <- (q_l * W_l) mod p_l  (an 8-bit x 8-bit product; widths: ByteOperands)
Mul == /\ stage = "mul"
       /\ y' = IF UseQ THEN (Q[l] * w[l]) % Moduli[l] ELSE w[l]
       /\ stage' = "mac" /\ k' = 1 /\ c' = 0
       /\ UNCHANGED <<w, l, x, ovf, peak>>

\* x <- x + M_l * y: limb k of the fused multiply-add carry chain.
Mac == /\ stage = "mac"
       /\ LET t     == x[k] + MhL[l][k] * y + c
              carry == t \div 2^LBits[k]
          IN /\ x' = [x EXCEPT ![k] = t % 2^LBits[k]]
             /\ IF k < Limbs
                THEN /\ k' = k + 1
                     /\ c' = IF LimbBug = "mac_nocarry" THEN 0 ELSE carry
                     /\ UNCHANGED <<stage, ovf, peak>>
                ELSE \* top limb: the carry-out leaves the word (unsigned wrap)
                     /\ ovf' = (ovf \/ carry # 0)
                     /\ peak' = Max(peak, XWord + Mh[l] * y)   \* ghost: exact sum
                     /\ stage' = "cmp" /\ k' = Limbs /\ c' = 0
       /\ UNCHANGED <<w, l, y>>

\* Compare x with P (step l <= N) or with H (centering, l = N+1), one limb
\* at a time from the top; then branch.
Cmp == /\ stage = "cmp"
       /\ LET K       == IF l <= N THEN PL ELSE HL
              gt      == x[k] > K[k]
              lt      == x[k] < K[k]
              decided == gt \/ lt \/ k = 1 \/ LimbBug = "cmp_toponly"
              r       == IF gt THEN "gt" ELSE IF lt THEN "lt" ELSE "eq"
              \* step l: x >= P ?  centering: x > H (or x >= H if not inclusive)
              doSub   == IF l <= N THEN UseCondSub /\ r # "lt"
                         ELSE IF CenterInclusive THEN r = "gt" ELSE r # "lt"
          IN IF ~decided
             THEN /\ k' = k - 1
                  /\ UNCHANGED <<w, l, stage, y, x, c, ovf, peak>>
             ELSE IF doSub
             THEN /\ stage' = "sub" /\ k' = 1 /\ c' = 0
                  /\ UNCHANGED <<w, l, y, x, ovf, peak>>
             ELSE IF l <= N
             THEN /\ NextHead
                  /\ UNCHANGED <<w, y, x, ovf, peak>>
             ELSE \* z = x >= 0: the sign bit must be clear
                  /\ stage' = "done" /\ k' = 1 /\ c' = 0
                  /\ ovf' = (ovf \/ SignBit(x) # 0)
                  /\ UNCHANGED <<w, l, y, x, peak>>

\* x <- x - P: limb k of the borrow chain.
Sub == /\ stage = "sub"
       /\ LET d      == x[k] - PL[k] - c
              borrow == IF d < 0 THEN 1 ELSE 0
              x2     == [x EXCEPT ![k] = d + borrow * 2^LBits[k]]
          IN /\ x' = x2
             /\ IF k < Limbs
                THEN /\ k' = k + 1
                     /\ c' = IF LimbBug = "sub_noborrow" THEN 0 ELSE borrow
                     /\ UNCHANGED <<w, l, stage, y, ovf, peak>>
                ELSE IF l <= N   \* top limb; the borrow-out leaves the word
                THEN /\ NextHead
                     /\ UNCHANGED <<w, y, ovf, peak>>
                ELSE \* z = x - P in two's complement: the sign bit must equal
                     \* the sign of the exact difference, i.e. the borrow-out
                     /\ stage' = "done" /\ k' = 1 /\ c' = 0
                     /\ ovf' = (ovf \/ SignBit(x2) # borrow)
                     /\ UNCHANGED <<w, l, y, peak>>

Next == Mul \/ Mac \/ Cmp \/ Sub
\* Terminates at "done" (the cfgs run without deadlock checking, as in check.sh).
Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

-----------------------------------------------------------------------------
(* Refinement mappings and the instances *)

phaseBar == IF stage = "done" THEN "done" ELSE "ready"
zBar     == IF stage = "done" THEN Signed(XWord) ELSE 0

Abs == INSTANCE CRTExact WITH Moduli <- Moduli, w <- w, phase <- phaseBar, z <- zBar
W   == INSTANCE CRTWord  WITH Moduli <- Moduli, N <- N, P <- P, H <- H, Mh <- Mh,
                              Q <- Q, ResBits <- ResBits, WordBits <- WordBits,
                              UseQ <- UseQ, UseCondSub <- UseCondSub,
                              CenterInclusive <- CenterInclusive,
                              w <- w, l <- l, stage <- StageW, y <- y, x <- XWord,
                              ovf <- ovf

-----------------------------------------------------------------------------
(* Invariants *)

TypeOK == /\ w \in Inputs
          /\ l \in 1..(N + 1)
          /\ stage \in {"mul", "mac", "cmp", "sub", "done"}
          /\ y \in 0..(MaxMod - 1)
          /\ x \in [LK -> Nat] /\ \A j \in LK : x[j] < 2^LBits[j]   \* limbs in range
          /\ k \in LK
          /\ c \in Nat
          /\ ovf \in BOOLEAN
          /\ peak \in Nat
          /\ W!TypeOK                       \* the mapped word is a WordBits-bit value

\* Limb arithmetic widths (real: 32-bit limbs, 8-bit y). The MAC carry is
\* < 2^ResBits, so the per-limb accumulator x_k + M_l,k*y + c < 2^(b+ResBits)
\* (40 bits real: fits a 64-bit register). A borrow is one bit.
MacCarryBound == stage = "mac" => c < 2^ResBits
BorrowBit     == stage = "sub" => c \in {0, 1}

\* Word-level claims of §3.6, evaluated on the mapped word (see CRTWord).
ByteOperands   == W!ByteOperands
AddendBound    == W!AddendBound
XBelowP        == W!XBelowP
RunningCRT     == W!RunningCRT
NoWordOverflow == W!NoWordOverflow

\* The closed form PeakExact bounds every intermediate x + M_l*y.
PeakBound == peak <= PeakExact
\* Sentinel, used only in neg_peak_attained: its violation shows PeakExact is
\* actually reached, so the bound is exact (and so is the certificate).
PeakBelowFormula == peak < PeakExact

\* §3.6 centering: the result lies in the unique-representation range.
ZInRange   == stage = "done" => Signed(XWord) \in ZRange
\* ... and is congruent to every residue plane.
ZCongruent == stage = "done" => Abs!IsCRT(Signed(XWord), w)
\* §3.6 "z = (A'B') exactly": whatever the true product c is, if |c| <= H
\* (the §3.2 guarantee) and c has these residues, then z = c.
\* O(P) per final state: used in the small (8753) models only.
ZExact == stage = "done" =>
            \A cc \in (-H)..H : [i \in Ls |-> cc % Moduli[i]] = w => Signed(XWord) = cc
\* The abstract step is deterministic: exactly one admissible z per input
\* (guards IsCRT against weakening; O(P) per final state, 8753 only).
AbsDeterministic == stage = "done" =>
            Cardinality({cc \in ZRange : Abs!IsCRT(cc, w)}) = 1

-----------------------------------------------------------------------------
(* Properties *)

\* §3.6 [spec: CRTRealization]: the limb realization refines exact CRT,
\* including the abstract fairness (the result is eventually produced).
Refines           == Abs!Spec
RefinesSafety     == Abs!SafetySpec          \* no liveness tableau (big models)
\* ... and refines the word-level §3.6 pseudocode, micro-step by micro-step.
RefinesWord       == W!Spec
RefinesWordSafety == W!SafetySpec
Termination       == <>(stage = "done")

=============================================================================
