------------------------------ MODULE IndexMap ------------------------------
(***************************************************************************)
(* IndexMap lowering and legality (DESIGN §5, §1.1, the K/N padding of     *)
(* §4/§4a and the read/write rules of §3.7 for k = 0 and empty I/J/L).     *)
(*                                                                         *)
(* A STATIC, VALUE-AGNOSTIC model. A problem is a descriptor triple        *)
(* (A, B, C): per tensor a base offset and per mode a label, an extent and *)
(* a stride (in elements; signed; zero allowed). No tensor VALUES are      *)
(* modelled: a contraction is fully described by the multiset of           *)
(* contribution triples                                                    *)
(*        <<addrC, addrA, addrB>>   meaning   C[addrC] += A[addrA]*B[addrB] *)
(* together with the set of C addresses it writes (each must be written   *)
(* exactly once: C <- alpha*sum + beta*C). Two procedures with equal       *)
(* contribution multisets and equal write sets compute the same exact C   *)
(* for EVERY choice of values, alpha and beta, so comparing them is the   *)
(* strongest value-free equivalence.                                       *)
(*                                                                         *)
(* REFERENCE (einsum, DESIGN §1): with ext(l) the extent of label l and x *)
(* ranging over all assignments label -> 0..ext(l)-1,                      *)
(*     addr_X(x) = base_X + sum_{modes p of X} s_X[p] * x[label_X[p]]      *)
(*     Einsum    = Bag{ <<addr_C(x), addr_A(x), addr_B(x)>> : x }          *)
(* A label repeated inside one operand uses the same index in every mode   *)
(* (diagonal); a label only in A (or only in B) is summed; a label only in *)
(* C is broadcast (v1 rejects it, see RuleCOnly).                          *)
(*                                                                         *)
(* IMPLEMENTATION MODEL (DESIGN §5 steps 1-10, §4 K2/K3/K4/K5, §4a, §6):  *)
(*  status   plan time (imd_plan_create), first failing row wins:          *)
(*           1 extents (INVALID_VALUE), 2 repeated C label                 *)
(*           (NOT_SUPPORTED), 3 C-only label with extent > 1              *)
(*           (NOT_SUPPORTED), 4 the sorted-|stride| rule on a non-empty C  *)
(*           (INVALID_VALUE), 5 the CUTLASS GemmCoord limit m, nh <=       *)
(*           GemmMax when the emulated GEMM runs (EMULATE, non-empty       *)
(*           output, k >= 2), on the POST-swap m and nh (INVALID_VALUE);   *)
(*           then the §3.2 RN accuracy policy: kappa = ceil(sqrt(k)) >= T  *)
(*           -> NOT_SUPPORTED, only when the emulated GEMM runs (EMULATE,  *)
(*           non-empty output, k >= 2) with rounding RN (T = T(N) is an    *)
(*           option of the call, the Execs field tn);                      *)
(*           then execute time                                             *)
(*           (imd_contract, DESIGN §6 order): null pointer             *)
(*           (INVALID_VALUE; C always, A/B only when alpha # 0 and k # 0,  *)
(*           ws only when workspace_size > 0), empty output (SUCCESS,      *)
(*           nothing touched),                                             *)
(*           ALIASING (C vs A/B byte hulls; workspace [ws, ws +            *)
(*           workspace_size) vs A/B/C), MISALIGNED (A/B/C not 8-byte,      *)
(*           ws not AlignWS-aligned), INSUFFICIENT_WORKSPACE.              *)
(*  swap     §5 step 8 coalescing: among C's modes with extent > 1, if the *)
(*           smallest-|stride| one carries an I label, the operand roles   *)
(*           are swapped (Cᵀ = BᵀAᵀ: the descriptors A and B exchange      *)
(*           places, so I and J exchange) BEFORE everything that uses m,  *)
(*           n: the §4a sizing, the GEMM limits, the lowering, and the     *)
(*           canonical orders (post-swap roles; the position tie-break     *)
(*           uses the CALLER's label order A.l ++ B.l ++ C.l).             *)
(*  lower    on the (possibly swapped) descriptor: step 2 diagonal merge   *)
(*           (strides of a repeated label summed: MS below), step 4        *)
(*           extent-1 labels dropped, step 5/6 classification by           *)
(*           membership (A-only/B-only -> K with stride 0 in the other     *)
(*           operand), step 8 canonical order per group, step 9 fusion,    *)
(*           step 10 additive scatter vectors                              *)
(*              off_X(lam, row, col) = base_X + v_X,L[lam] + v_X,G1[row]   *)
(*                                     + v_X,G2[col]                       *)
(*           with (G1, G2) = (I, K) for A, (K, J) for B, (I, J) for C.     *)
(*  gather   K2/K3: A'[lam][i][h] at (lam*m + i)*kh + h, kh = ceil_Q(k);   *)
(*           B'[lam][j][h] at (lam*nh + j)*kh + h, nh = ceil_Q(n). Cells    *)
(*           with h >= k (and rows j >= n of B') are PAD (value 0). These  *)
(*           are exactly the §4a regions RA, RB (per modulus): b*m*kh and  *)
(*           b*nh*kh cells.                                                *)
(*  gemm     K4: every (lam, i, j < nh, h < kh) multiplies A' and B'       *)
(*           cells; products with a PAD factor are 0.                      *)
(*  scatter  K5: plane cell (lam*m + i)*nh + j, for j < n only, goes to    *)
(*           off_C(lam, i, j). Columns j >= n are never scattered.         *)
(*  plan     dispatch by descriptor (§4, §4a): NOOP if b*m*n = 0 (nothing  *)
(*           read or written), SCALE if k = 0 (C <- beta*C: every C        *)
(*           element written, A and B not read), DIRECT if mode = DIRECT   *)
(*           or K is empty (no K label after step 4; k = 1): native sum    *)
(*           over the canonical k-bar order, no GEMM; GEMM otherwise.      *)
(*  sizing   §4a per-region element counts and workspace_size in bytes     *)
(*           (real widths, one modulus plane: N = g = 1; Pipeline checks   *)
(*           the N/g structure), from the post-swap m, n, k, b.            *)
(*                                                                         *)
(* CLAIMS (invariants; each comment names the DESIGN statement it checks): *)
(*  Partition, EinsumPreserved, ScatterExactlyOnce, ScatterCoversC,        *)
(*  AccessInBounds, ReadsExact, ZeroExtentRules, PaddingInert,             *)
(*  LayoutBijective, DecodeBijective, FusionPreservesAddresses,            *)
(*  OperandIndependence, NoCAliasing, StrideRuleImpliesInjective,          *)
(*  StatusCorrect (incl. the full plan + contract precedence chain),       *)
(*  CanonicalOrderSorted, CanonicalOrderExact (the full key, post-swap     *)
(*  roles, caller's position), OrderIndependent (expensive; small models), *)
(*  the role swap: SwapWellDefined, SwapMakesJFastest (and every claim     *)
(*  above holds on the swapped lowering), the sizing and dispatch:         *)
(*  DispatchByDescriptor, WorkspaceFits, WorkspaceExact, GemmDimsFit,      *)
(*  LimitScope, the accuracy policy: KappaAccepted, KappaScope, and the   *)
(*  execute-time checks: NoNullDeref, WsNoAlias,                           *)
(*  AliasingSound, AlignedAccess, WsSufficient; and                        *)
(*  StrideRuleNecessary (FALSE by design: a negative model shows the rule  *)
(*  is only sufficient).                                                   *)
(*                                                                         *)
(* COVERAGE. Every descriptor of a model is enumerated as an initial state *)
(* (Init); the derived status, lowering and einsum are stored in state     *)
(* variables so each is computed once. MC_IndexMap ASSUMEs that each      *)
(* positive model enumerates and accepts the named boundary descriptors   *)
(* (MustCover), and pins every negative model's failure to a witness      *)
(* descriptor that the design (Drop = "none") handles correctly.           *)
(*                                                                         *)
(* SCALING. Real descriptors have rank <= 64, int64 extents and strides,   *)
(* and pad quantum 16 (CUTLASS int8 alignment). The models use ranks <= 3, *)
(* extents in 0..3, strides in about -3..4, a few labels and PadQ = 2 or  *)
(* 4. Nothing in the spec depends on a width: every rule is a statement   *)
(* about linear address maps over finite boxes. Extents 0 and 1 are the   *)
(* empty and dropped behaviours; a real mode needs BOTH extent 2 and       *)
(* extent 3 models: at 2 every (e-1) factor is 1 and every k, n is a power *)
(* of two, so the (e-1) of the stride rule and of the interval test, k     *)
(* and n that are not multiples of PadQ, and mixed radices (fused 2*3,     *)
(* 3*2) are only exercised at 3 (pos_ext3, pos_alias3). The int64/int32   *)
(* overflow guards of the §5 table are NOT modelled (closed-form range     *)
(* checks; see cert_claims).                                               *)
(*                                                                         *)
(* NOT COVERED HERE: the alpha = 0 / beta = 0 read rules of §3.7 (A, B not *)
(* read when alpha = 0; C not read when beta = 0) are Pipeline's (DESIGN   *)
(* §4 "[spec: Pipeline]"); this module models only the zero-EXTENT rules.  *)
(*                                                                         *)
(* DESIGN RULES that a negative model drops (Drop constant):               *)
(*  "Extents"    §5 step 1      extent consistency per label               *)
(*  "RepeatC"    §5 step 3      repeated label within C -> NOT_SUPPORTED   *)
(*  "CStride"    §5 table       sorted-|stride| rule on C                  *)
(*  "Aliasing"   §5 table       C vs A/B byte-interval test                *)
(*  "Single"     §5 step 6      A-only/B-only label -> K, stride 0 in the  *)
(*                              other operand (dropped: the label is left  *)
(*                              out of every group, as TBLIS would)        *)
(*  "Fusion"     §5 step 9      fusion condition checked in EVERY tensor   *)
(*                              carrying the group (dropped: only in the   *)
(*                              tensors where both labels appear, which is *)
(*                              the literal v1 wording)                    *)
(*  "COnly"      §5 step 7      C-only label with extent > 1 rejected      *)
(*                              (dropping it is a POSITIVE model: the rule *)
(*                              is a v1 policy, broadcasting is correct)   *)
(*  "NoSwap"     §5 step 8      the coalescing role swap is not done       *)
(*  "SwapFilter" §5 step 8      the swap looks at ALL of C's modes, not    *)
(*                              only those with extent > 1                 *)
(*  "SwapLate"   §5 step 8      the swap happens AFTER sizing: §4a sizes   *)
(*                              and the GEMM limits use the pre-swap m, n  *)
(*                              (the lowering itself still swaps)          *)
(*  "GemmLimit"  §5 row 5       m, nh <= GemmMax not checked               *)
(*  "GemmLimitN"      row 5     checks n instead of nh = ceil_Q(n)         *)
(*  "GemmLimitDirect" row 5     also applied in DIRECT mode                *)
(*  "GemmLimitEmpty"  row 5     also applied to an empty output            *)
(*  "GemmLimitK"      row 5     also applied when k < 2 (no GEMM runs)     *)
(*  "Kappa"      §3.2/§6        the kappa >= T policy not applied          *)
(*  "KappaDirect"     §3.2      also applied in DIRECT mode                *)
(*  "KappaEmpty"      §3.2      also applied to an empty output            *)
(*  "KappaK"          §3.2      also applied when k < 2 (k = 1: K-empty -> *)
(*                              DIRECT; needs T <= 1, see KappaScope)      *)
(*  "KappaTrunc"      §3.2      also applied with rounding TRUNC           *)
(*  "OrderPreSwap" §5 step 8    canonical orders computed BEFORE the swap  *)
(*                              (K's "big" on the caller's roles; differs  *)
(*                              only when m = n)                           *)
(*  "PosSwapped"   §5 step 8    position tie-break from the swapped        *)
(*                              descriptor (B.l ++ A.l ++ C.l)             *)
(*  "Null"       §6 1           null pointers not rejected                 *)
(*  "NullStrict" §6 1           any null pointer rejected (the pre-M1      *)
(*                              literal rule: A/B even when alpha = 0 or   *)
(*                              k = 0, ws even when workspace_size = 0)    *)
(*  "NullWsLax"  §6 1           null ws accepted even when                 *)
(*                              workspace_size > 0                         *)
(*  "WsAlias"    §5 row 6       workspace vs A/B/C test dropped            *)
(*  "WsRangeBytes" row 6        workspace range [ws, ws + ws_bytes)        *)
(*                              instead of [ws, ws + workspace_size)       *)
(*  "Misaligned" §5 row 7       alignment test dropped                     *)
(*  "Insufficient" §6 5         ws_bytes >= workspace_size not checked     *)
(* PRECEDENCE variants (the contract checks reordered; StatusCorrect, a   *)
(* table written independently of the implementation chain, fails):      *)
(*  "OrderNullEmpty"   empty output before the null check                 *)
(*  "OrderEmptyLast"   empty output after INSUFFICIENT_WORKSPACE          *)
(*  "OrderAlignFirst"  MISALIGNED before ALIASING                         *)
(*  "OrderInsuffFirst" INSUFFICIENT_WORKSPACE before MISALIGNED           *)
(*                                                                         *)
(* D1 PACKED OPERANDS (DESIGN §6b.2, §6b.5; constants Packing, PackEnum;  *)
(* "none" in every model above). A pack of the caller's A or B is built   *)
(* on that operand's descriptor (same labels, extents, base; its own       *)
(* strides: X's, or with PackEnum "strides" any of the model's, i.e. a     *)
(* different tensor), with its own classes K, L, R and any order per      *)
(* class (PackEnum "plan": the plan's classes, which imd_pack_desc_for     *)
(* fills; "classes": any partition), which covers GIVEN and AUTO. Packs    *)
(* that imd_pack_layout rejects (K empty, κ >= T with RN) are not          *)
(* enumerated. The plan (imd_plan_create_packed) ADOPTS the pack's order  *)
(* for K, L and X's own row class (I for post-swap A, J for B; K, L from  *)
(* post-swap A's pack when both are packed), keeps v1.5's for the         *)
(* partner's rows, fuses in the adopted order (all carriers), computes    *)
(* X's slot key from that lowering restricted to X (X's strides: a       *)
(* partner-only K label is a mode (e, 0)) and compares it with the pack's  *)
(* key word for word: b, r, k, then for L, R, K the modes fused over X's  *)
(* strides alone. Status: §5 rows 1-5, then PACK_MISMATCH (a key differs, *)
(* a partner-only K label: _CLASSES, both packed with different K or L    *)
(* orders: _ORDER), then κ >= T. The GEMM reads X FROM THE PACK (its       *)
(* §6b.4 image as element addresses: (λ·r + ρ)·k̂ + h, K padding 0, then a  *)
(* zeroed tail of (⌈r⌉ − r)·k̂): post-swap A at (λ·m + i)·k̂ + h, post-swap *)
(* B rows j < n̂ at batch stride r·k̂ = n·k̂ (rows j >= n: the next block or *)
(* the tail, finite, in W columns K5 never reads). §4a's sX, fX, RX of a  *)
(* packed role are 0. imd_contract_packed's execute-time checks are not    *)
(* modelled (packed models use XDef-like arguments).                       *)
(* Claims: every claim above on the packed lowering (EinsumPreserved etc., *)
(* for every enumerated order), PackKeySound (equal keys => the pack's     *)
(* image IS the plan's operand buffer), PackComplete (a pack with the      *)
(* plan's classes in any order is accepted), DescForReproduces,            *)
(* AdoptedOrderExact, AutoOrderV15, PackNoOOB, PackPaddingInert,           *)
(* PackPolicy, KeyRejectsPartnerOnly (except k = 0).                       *)
(* D1 rules a negative model drops:                                        *)
(*  "PackFuseDefault"  fusion runs of the v1.5 order applied to the       *)
(*                     adopted order (fusion not recomputed)               *)
(*  "PackNoAdopt"      the plan keeps v1.5's orders (key and lowering)     *)
(*  "PackNoAdoptLower" the key uses the adopted order, the lowering not    *)
(*  "KeyNoStrides"     the key without strides (extents only)              *)
(*  "KeyFuseCarriers"  the slot key fused over all carriers, not X alone  *)
(*  "PackAfterKappa"   κ >= T checked before PACK_MISMATCH                 *)
(*  "PackBStrideNh"    role B read at v1's batch stride n̂·k̂               *)
(*  "PackNoTail"       packs without the zeroed tail                       *)
(*  "PackPartnerOnly"  no partner-only policy (POSITIVE: the key suffices) *)
(*  "PackOrderCheck"   no _ORDER check (POSITIVE: the key suffices)        *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, Bags, TLC

CONSTANTS
    Labels,          \* 1..NL, label ids
    RankA, RankB, RankC,   \* maximal rank enumerated per tensor
    ExtentSet,       \* extents enumerated (subset of 0..2 in the models)
    PerModeExtents,  \* TRUE: every mode picks its own extent (mismatches
                     \*   possible); FALSE: one extent per label
    StridesA(_), StridesB(_), StridesC(_),
                     \* r |-> set of stride sequences of length r
    BasesA, BasesB, BasesC,  \* base offsets enumerated
    PadQ,            \* pad quantum of kh, nh (real: 16)
    CanonicalLabels, \* TRUE: enumerate one labelling per renaming class
    Drop,            \* "none" or the ONE design rule this model drops
    CheckOrders,     \* TRUE: also compute OrderIndependent (expensive)
    MustCover,       \* names of coverage descriptors (defined in MC) that
                     \*   this model must enumerate AND accept (ASSUME in MC)
    Execs,           \* set of execute-time argument records (see XDef)
    GemmMax,         \* CUTLASS GemmCoord limit on m and nh (real 2^31 - 1)
    AlignWS,         \* workspace region alignment (real 256 bytes)
    Packing,         \* D1: the caller's operands packed: "none" | "A" | "B" | "AB"
    PackEnum         \* D1: the packs enumerated: "plan" (the plan's classes,
                     \*   every order per class), "classes" (every partition of
                     \*   X's labels into K, L, R, every order), "strides" ("plan"
                     \*   plus every stride vector of the model for the packed
                     \*   tensor: a different tensor's pack), "all"

VARIABLES
    desc,   \* the descriptor triple (chosen in Init, then constant)
    st,     \* its status
    low,    \* the lowering (a record), or "none" if rejected
    ein,    \* the einsum reference (a record), or "none" if rejected
    el,     \* el[X] = Elems(desc, X), cached
    xa,     \* the plan option and imd_contract arguments (an Execs record)
    pk      \* D1: the packs the plan is created with (NoPk: imd_plan_create)

vars == <<desc, st, low, ein, el, xa, pk>>

-----------------------------------------------------------------------------
(* Helpers *)

Tens  == {"A", "B", "C"}
Grps  == {"I", "J", "K", "L"}

\* Tensors that carry each group in the GEMM view (DESIGN §1.1).
Car   == [I |-> {"A", "C"}, J |-> {"B", "C"}, K |-> {"A", "B"},
          L |-> {"A", "B", "C"}]

Abs(x)    == IF x < 0 THEN -x ELSE x
Min(x, y) == IF x < y THEN x ELSE y
Max(x, y) == IF x > y THEN x ELSE y
SetMax(S) == CHOOSE x \in S : \A y \in S : y <= x

\* Sum / product of f[x] over a finite set S (f's domain contains S).
RECURSIVE SumOver(_, _)
SumOver(S, f) == IF S = {} THEN 0
                 ELSE LET x == CHOOSE x \in S : TRUE
                      IN  f[x] + SumOver(S \ {x}, f)
RECURSIVE ProdOver(_, _)
ProdOver(S, f) == IF S = {} THEN 1
                  ELSE LET x == CHOOSE x \in S : TRUE
                       IN  f[x] * ProdOver(S \ {x}, f)

CeilQ(x)  == ((x + PadQ - 1) \div PadQ) * PadQ

LexLt(a, b) == \E t \in DOMAIN a :
                  a[t] < b[t] /\ \A u \in 1..(t - 1) : a[u] = b[u]

\* The elements of S listed in increasing key order (keys distinct).
SortBy(S, key) ==
    [t \in 1..Cardinality(S) |->
        CHOOSE x \in S : Cardinality({y \in S : LexLt(key[y], key[x])}) = t - 1]

\* A Bags-module bag: element -> multiplicity, from the family F over P.
MkBag(P, F) == [t \in {F[p] : p \in P} |-> Cardinality({p \in P : F[p] = t})]

\* Largest index value a descriptor needs: from ITS OWN extents, not from
\* ExtentSet, so the ASSUMEs in MC_IndexMap may evaluate descriptors whose
\* extents lie outside the enumerated ExtentSet (e.g. extent 3 witnesses in
\* an extent-2 model) without truncating their index boxes.
MaxIdx(d) == SetMax(UNION {{d[X].e[p] : p \in 1..Len(d[X].e)} : X \in Tens} \cup {1}) - 1

\* Address sentinel for a nonzero product that lands in an unscattered
\* (padding) column of the u8 plane. Must never appear (EinsumPreserved).
Unscattered == -1000000

Pad   == <<TRUE, 0>>
Cell(a) == <<FALSE, a>>

-----------------------------------------------------------------------------
(* Descriptor accessors. d[X] = [base, l, e, s] with l, e, s sequences.  *)

Rk(d, X)  == Len(d[X].l)
Pos(d, X) == 1..Rk(d, X)
Lab(d, X) == {d[X].l[p] : p \in Pos(d, X)}
AllLab(d) == Lab(d, "A") \cup Lab(d, "B") \cup Lab(d, "C")
Occ(d)    == UNION {{<<X, p>> : p \in Pos(d, X)} : X \in Tens}
OccLab(d, o) == d[o[1]].l[o[2]]
OccExt(d, o) == d[o[1]].e[o[2]]

\* §5 step 1.
ExtentsConsistent(d) ==
    \A o1, o2 \in Occ(d) : OccLab(d, o1) = OccLab(d, o2) => OccExt(d, o1) = OccExt(d, o2)

\* The extent the lowering uses. Unique when ExtentsConsistent. When the
\* extent rule is dropped (negative model) the largest one is used.
Ext(d, l) == SetMax({OccExt(d, o) : o \in {o \in Occ(d) : OccLab(d, o) = l}})

\* §5 step 2: merged (diagonal) stride of label l in tensor X; 0 if absent
\* (this 0 IS the stride-0 lowering of step 6 for the other operand).
MS(d, X, l) == SumOver({p \in Pos(d, X) : d[X].l[p] = l}, d[X].s)

RepeatedC(d) == \E p, q \in Pos(d, "C") : p # q /\ d.C.l[p] = d.C.l[q]
COnly(d)     == Lab(d, "C") \ (Lab(d, "A") \cup Lab(d, "B"))
COnlyBig(d)  == \E l \in COnly(d) : Ext(d, l) > 1
\* The NOT_SUPPORTED reasons this model enforces.
Unsupported(d) == RepeatedC(d) \/ (Drop # "COnly" /\ COnlyBig(d))

\* Per-tensor element set, from the tensor's own modes (its own extents).
ModeIdx(d, X) == {y \in [Pos(d, X) -> 0..MaxIdx(d)] : \A p \in Pos(d, X) : y[p] < d[X].e[p]}
AddrM(d, X, y) == d[X].base + SumOver(Pos(d, X), [p \in Pos(d, X) |-> d[X].s[p] * y[p]])
Elems(d, X)   == {AddrM(d, X, y) : y \in ModeIdx(d, X)}
Empty(d, X)   == \E p \in Pos(d, X) : d[X].e[p] = 0
Injective(d, X) == Cardinality(Elems(d, X)) = Cardinality(ModeIdx(d, X))

\* §5 table, C self-overlap: over modes with extent > 1, sorted by |stride|
\* (ties in mode order), |s_(t)| > sum_{u<t} |s_(u)|*(e_(u)-1).
StrideRule(d, X) ==
    LET e == d[X].e  s == d[X].s
        Big == {p \in Pos(d, X) : e[p] > 1}
        span == [p \in Pos(d, X) |-> Abs(s[p]) * (e[p] - 1)]
    IN  \A t \in Big :
           Abs(s[t]) > SumOver({u \in Big \ {t} : Abs(s[u]) < Abs(s[t])
                                     \/ (Abs(s[u]) = Abs(s[t]) /\ u < t)}, span)

\* §5 table, C vs A / C vs B: half-open element interval [lo, hi); empty
\* (lo = hi) when the tensor has a zero extent. (Real: bytes = 8 * elements,
\* pointers 8-byte aligned, so the element test is the byte test.)
Interval(d, X) ==
    IF Empty(d, X) THEN <<0, 0>>
    ELSE LET e == d[X].e  s == d[X].s IN
         <<d[X].base + SumOver(Pos(d, X), [p \in Pos(d, X) |-> Min(0, (e[p] - 1) * s[p])]),
           d[X].base + SumOver(Pos(d, X), [p \in Pos(d, X) |-> Max(0, (e[p] - 1) * s[p])]) + 1>>
Intersect(i1, i2) == i1[1] < i1[2] /\ i2[1] < i2[2] /\ i1[1] < i2[2] /\ i2[1] < i1[2]
IntervalAlias(d) == Intersect(Interval(d, "C"), Interval(d, "A"))
                 \/ Intersect(Interval(d, "C"), Interval(d, "B"))

-----------------------------------------------------------------------------
(* Lowering (DESIGN §5 steps 2-10).                                       *)

\* §5 steps 5-7: class by membership set.
ClassOf(d, l) ==
    LET M == {X \in Tens : l \in Lab(d, X)} IN
    CASE M = {"A", "C"}      -> "I"
      [] M = {"B", "C"}      -> "J"
      [] M = {"A", "B"}      -> "K"
      [] M = Tens            -> "L"
      [] M = {"A"} \/ M = {"B"} -> IF Drop = "Single" THEN "none" ELSE "K"
      \* Only in C. Extent 1 was dropped (step 4), extent > 1 rejected (step
      \* 7); what remains is extent 0, i.e. C is empty. Lowered to I with
      \* stride 0 in A, so m = 0 and the plan is a NOOP (design_issues).
      [] M = {"C"}           -> "I"

\* Position of l's first occurrence in A.l \o B.l \o C.l: the final, label-
\* id-free tie-break of the canonical order (keeps the order a function of
\* the descriptors only, so CanonicalLabels is a true symmetry reduction).
FirstPos(d, l) ==
    LET cat == d.A.l \o d.B.l \o d.C.l
    IN  CHOOSE p \in 1..Len(cat) : cat[p] = l /\ \A q \in 1..(p - 1) : cat[q] # l

\* Mixed-radix vector over a mode list (first mode fastest):
\*   v[q] = sum_j modes[j].s[X] * ((q \div prod_{u<j} modes[u].e) % modes[j].e)
RadixVec(modes, X, size) ==
    LET n   == Len(modes)
        pre == [j \in 1..n |-> ProdOver(1..(j - 1), [u \in 1..n |-> modes[u].e])]
    IN  [q \in 0..(size - 1) |->
           SumOver(1..n, [j \in 1..n |-> modes[j].s[X] * ((q \div pre[j]) % modes[j].e)])]

\* Decode of canonical index q of an ordered label list (first fastest).
Decode(o, ext, size) ==
    LET n   == Len(o)
        pre == [j \in 1..n |-> ProdOver(1..(j - 1), [u \in 1..n |-> ext[o[u]]])]
    IN  [q \in 0..(size - 1) |->
           [l \in {o[j] : j \in 1..n} |->
              LET j == CHOOSE j \in 1..n : o[j] = l IN (q \div pre[j]) % ext[l]]]

\* D1 packed operands (DESIGN §6b). A pack is built on the CALLER's operand
\* op (same labels, extents and base; its own strides s, which are op's own
\* strides unless the model enumerates "a different tensor's pack"), with
\* its own classes and canonical orders (label sequences): K (contracted),
\* L (batch), R (rows). pk = [A |-> pack or NoPack, B |-> pack or NoPack],
\* indexed by the caller's operand.
NoPack == [none |-> TRUE]   \* "no pack" (a record: TLC compares records with records)
IsPk(p) == "op" \in DOMAIN p
NoPk == [A |-> NoPack, B |-> NoPack]
SeqSet(sq) == {sq[i] : i \in DOMAIN sq}

\* Fusion over ONE tensor's strides (the key's "fused over X's strides
\* alone"): consecutive modes <<e, s>>, <<e', s'>> merge iff s' = s*e.
FuseModes(ms) ==
    LET n   == Len(ms)
        S   == {t \in 1..n : t = 1 \/ ms[t][2] # ms[t - 1][2] * ms[t - 1][1]}
        st0 == SortBy(S, [t \in S |-> <<t>>])
        ns  == Len(st0)
        last(j) == IF j < ns THEN st0[j + 1] - 1 ELSE n
    IN  [j \in 1..ns |-> <<ProdOver(st0[j]..last(j), [t \in 1..n |-> ms[t][1]]), ms[st0[j]][2]>>]

\* Mixed-radix vector over modes <<e, s>> (first fastest).
RadixS(modes, size) ==
    LET n   == Len(modes)
        pre == [j \in 1..n |-> ProdOver(1..(j - 1), [u \in 1..n |-> modes[u][1]])]
    IN  [q \in 0..(size - 1) |->
           SumOver(1..n, [j \in 1..n |-> modes[j][2] * ((q \div pre[j]) % modes[j][1])])]

\* DESIGN §6b.2 compatibility key (FORMAT, N and rounding are equal by
\* construction here: PackedOperand checks those parts): b, r, k, then for
\* the classes L, R, K the fused modes (extent, stride in X) in canonical
\* order, fused over X's strides alone, extent-1 modes already dropped.
\* Drop "KeyNoStrides": the element map without the strides (extents only).
KMap(ms) == IF Drop = "KeyNoStrides" THEN [t \in DOMAIN ms |-> ms[t][1]] ELSE FuseModes(ms)
KeyOf(b, r, k, mL, mR, mK) == <<b, r, k, KMap(mL), KMap(mR), KMap(mK)>>

\* The pack of caller operand pck.op of descriptor d (imd_pack_layout + the
\* bytes imd_pack writes, as element addresses): σ-free view of the §6b.4
\* residue region of one plane: cell (λ·r + ρ)·k̂ + h holds X at (λ, ρ, h)
\* in the pack's enumeration (h < k), 0 for h >= k, then the zeroed tail of
\* (⌈r⌉ − r)·k̂ cells (Drop "PackNoTail": none). k̂ = ⌈k⌉_PadQ, ⌈r⌉ =
\* ⌈r⌉_PadQ (the GEMM's n̂ rounding).
PkMS(d, pck, l) == SumOver({p \in Pos(d, pck.op) : d[pck.op].l[p] = l}, pck.s)
PkInfo(d, pck) ==
    LET op  == pck.op
        ext == [l \in Lab(d, op) |-> Ext(d, l)]
        md(sq) == [t \in 1..Len(sq) |-> <<ext[sq[t]], PkMS(d, pck, sq[t])>>]
        sz(sq) == ProdOver(SeqSet(sq), ext)
        b == sz(pck.L)  r == sz(pck.R)  k == sz(pck.K)  kh == CeilQ(k)
        size == b * r * kh + (IF Drop = "PackNoTail" THEN 0 ELSE (CeilQ(r) - r) * kh)
        vL == RadixS(md(pck.L), b)  vR == RadixS(md(pck.R), r)  vK == RadixS(md(pck.K), k)
        buf == [q \in 0..(size - 1) |->
                  IF q < b * r * kh
                  THEN LET lam == q \div (r * kh)  rho == (q \div kh) % r  h == q % kh
                       IN  IF h < k THEN Cell(d[op].base + vL[lam] + vR[rho] + vK[h]) ELSE Pad
                  ELSE Pad]
    IN  [b |-> b, r |-> r, k |-> k, kh |-> kh, size |-> size, buf |-> buf,
         key |-> KeyOf(b, r, k, md(pck.L), md(pck.R), md(pck.K))]
\* A read of the pack at cell q; past its end: a junk non-pad cell (never an
\* address of anything, so EinsumPreserved fails as well as PackNoOOB).
OOBCell == <<FALSE, -2000000>>
PRead(buf, q) == IF q \in DOMAIN buf THEN buf[q] ELSE OOBCell

\* The canonical orders and fusion of the lowering (DESIGN §5 steps 2-9),
\* with D1's adopted orders, and the plan-time PACK_MISMATCH check. Cheap
\* (no buffers): the status chain uses it.
\* d is the descriptor the lowering works on (already swapped if sw); od is
\* the CALLER's descriptor. DESIGN §5 step 8 (as implemented in M1): the
\* swap comes first, the canonical orders are computed on the post-swap
\* roles (so K's "big" operand compares the post-swap m, n), and the
\* final tie-break "position" is taken from the caller's original label
\* order od.A.l \o od.B.l \o od.C.l.
\* D1 (§6b.2): a plan built against a pack of operand X ADOPTS the pack's
\* canonical order for the classes the pack carries: K, L and X's own row
\* class (I if X is post-swap A, J if post-swap B; the swap is still C's);
\* the partner's row class keeps the v1.5 order; with both operands packed
\* K and L come from the post-swap A's pack (the two must agree: _ORDER).
\* Step 9 fusion is then applied in the adopted order (all carriers). The
\* slot key of X is computed from this lowering restricted to X (X's
\* strides, so a partner-only K label is a mode (e, 0)), in the adopted
\* orders, and compared with the pack's key word for word.
OrdInfo(d, sw, od, pks) ==
    LET labs == AllLab(d)
        ext  == [l \in labs |-> Ext(d, l)]
        ms   == [X \in Tens, l \in labs |-> MS(d, X, l)]
        kept == {l \in labs : ext[l] # 1}                         \* step 4
        cls  == [l \in kept |-> ClassOf(d, l)]
        grp  == [g \in Grps |-> {l \in kept : cls[l] = g}]
        sz   == [g \in Grps |-> ProdOver(grp[g], ext)]
        m == sz["I"]  n == sz["J"]  k == sz["K"]  b == sz["L"]
        kh == CeilQ(k)  nh == CeilQ(n)
        \* step 8. Drop "OrderPreSwap": the order is computed BEFORE the
        \* swap, i.e. "big" is decided on the caller's roles (the caller's
        \* A = role B after a swap); this differs only when m = n.
        big == IF Drop = "OrderPreSwap" /\ sw
               THEN (IF n >= m THEN "B" ELSE "A")
               ELSE (IF m >= n THEN "A" ELSE "B")
        oth == IF big = "A" THEN "B" ELSE "A"
        \* Drop "PosSwapped": position taken from the swapped descriptor.
        posd == IF Drop = "PosSwapped" THEN d ELSE od
        key == [g \in Grps |-> [l \in grp[g] |->
                  LET a == Abs(ms["A", l])  bb == Abs(ms["B", l])
                      c == Abs(ms["C", l])  f == FirstPos(posd, l)
                  IN  CASE g = "I" -> <<c, a, 0, f>>
                        [] g = "J" -> <<c, bb, 0, f>>
                        [] g = "L" -> <<c, a, bb, f>>
                        [] g = "K" -> <<Abs(ms[big, l]), Abs(ms[oth, l]), 0, f>>]]
        ord15 == [g \in Grps |-> SortBy(grp[g], key[g])]          \* v1.5 orders
        \* D1 adoption: the pack of post-swap role R is the caller's cop(R)'s.
        cop(R) == IF sw THEN (IF R = "A" THEN "B" ELSE "A") ELSE R
        pR == [R \in {"A", "B"} |-> pks[cop(R)]]
        proles == {R \in {"A", "B"} : IsPk(pR[R])}
        kl == IF IsPk(pR["A"]) THEN pR["A"] ELSE pR["B"]
        adopt == [g \in Grps |-> CASE g \in {"K", "L"} -> kl
                                   [] g = "I"          -> pR["A"]
                                   [] g = "J"          -> pR["B"]]
        adopted == {g \in Grps : IsPk(adopt[g])}
        pseq(g) == IF g \in {"I", "J"} THEN adopt[g].R ELSE adopt[g][g]
        \* the pack's order of the class, then (only when the classes differ,
        \* i.e. the plan will be rejected) the rest in v1.5 order
        ordA == [g \in Grps |->
                   IF g \notin adopted THEN ord15[g]
                   ELSE SelectSeq(pseq(g), LAMBDA l : l \in grp[g])
                        \o SelectSeq(ord15[g], LAMBDA l : l \notin SeqSet(pseq(g)))]
        \* Drop "PackNoAdopt": the plan keeps v1.5's orders (key and lowering);
        \* "PackNoAdoptLower": the key uses the adopted orders, the lowering not.
        ordKey == IF Drop = "PackNoAdopt" THEN ord15 ELSE ordA
        ord    == IF Drop \in {"PackNoAdopt", "PackNoAdoptLower"} THEN ord15 ELSE ordA
        \* step 9: g_t fuses into g_(t-1) iff s_X(g_t) = s_X(g_(t-1))*e(g_(t-1))
        \* for every X in Chk.
        fuseOK(g, a, c) ==
            LET chk == IF Drop = "Fusion"
                       THEN {X \in Car[g] : a \in Lab(d, X) /\ c \in Lab(d, X)}
                       ELSE Car[g]
            IN  \A X \in chk : ms[X, c] = ms[X, a] * ext[a]
        starts(g, o) == {t \in 1..Len(o) : t = 1 \/ ~fuseOK(g, o[t - 1], o[t])}
        unf  == [g \in Grps |-> [t \in 1..Len(ord[g]) |->
                   [e |-> ext[ord[g][t]], s |-> [X \in Tens |-> ms[X, ord[g][t]]]]]]
        \* Drop "PackFuseDefault": the fusion runs are decided on the v1.5
        \* order and applied to the adopted one (fusion not recomputed).
        fused == [g \in Grps |->
                   LET o == ord[g]
                       S == IF Drop = "PackFuseDefault" THEN starts(g, ord15[g]) ELSE starts(g, o)
                       st0 == SortBy(S, [t \in S |-> <<t>>])
                       ns == Len(st0)
                       last(j) == IF j < ns THEN st0[j + 1] - 1 ELSE Len(o)
                   IN  [j \in 1..ns |->
                         [e |-> ProdOver(st0[j]..last(j), [t \in 1..Len(o) |-> ext[o[t]]]),
                          s |-> [X \in Tens |-> ms[X, o[st0[j]]]]]]]
        \* D1 slot key of post-swap role R (the plan's view of X = d[R]).
        rowG(R) == IF R = "A" THEN "I" ELSE "J"
        xm(R, g) == [t \in 1..Len(ordKey[g]) |-> <<ext[ordKey[g][t]], ms[R, ordKey[g][t]]>>]
        \* Drop "KeyFuseCarriers": the slot key takes the lowering's fusion over
        \* all carriers instead of fusing over X's strides alone.
        smap(R, g) == IF Drop = "KeyFuseCarriers"
                      THEN [j \in 1..Len(fused[g]) |-> <<fused[g][j].e, fused[g][j].s[R]>>]
                      ELSE KMap(xm(R, g))
        slotKey == [R \in proles |->
                      <<b, IF R = "A" THEN m ELSE n, k,
                        smap(R, "L"), smap(R, rowG(R)), smap(R, "K")>>]
        packKey == [R \in proles |-> PkInfo(od, pR[R]).key]
        keyEq   == [R \in proles |-> packKey[R] = slotKey[R]]
        \* partner-only K label: in the plan's K, not carried by X (§6b.10 #10)
        partnerOnly == [R \in proles |-> \E l \in grp["K"] : l \notin Lab(d, R)]
        orderAgree == proles = {"A", "B"} =>
                        pR["A"].K = pR["B"].K /\ pR["A"].L = pR["B"].L
        \* PACK_MISMATCH (§6b.5 imd_plan_create_packed): a slot key differs; a
        \* partner-only K label (_CLASSES, by policy; Drop "PackPartnerOnly"
        \* leaves it to the key); both packed with different K or L orders
        \* (_ORDER; Drop "PackOrderCheck" leaves it to the key).
        mis == \/ \E R \in proles : ~keyEq[R]
               \/ Drop # "PackPartnerOnly" /\ \E R \in proles : partnerOnly[R]
               \/ Drop # "PackOrderCheck" /\ ~orderAgree
    IN  [labs |-> labs, ext |-> ext, ms |-> ms, kept |-> kept, cls |-> cls, grp |-> grp,
         sz |-> sz, m |-> m, n |-> n, k |-> k, b |-> b, kh |-> kh, nh |-> nh,
         ord15 |-> ord15, ord |-> ord, unf |-> unf, fused |-> fused,
         pR |-> pR, proles |-> proles, adopted |-> adopted, slotKey |-> slotKey,
         packKey |-> packKey, keyEq |-> keyEq, partnerOnly |-> partnerOnly,
         orderAgree |-> orderAgree, mis |-> mis]

LowerP(d, mode, sw, od, pks) ==
    LET oi   == OrdInfo(d, sw, od, pks)
        labs == oi.labs  ext == oi.ext  grp == oi.grp  sz == oi.sz
        m == oi.m  n == oi.n  k == oi.k  b == oi.b  kh == oi.kh  nh == oi.nh
        ord == oi.ord  unf == oi.unf  fused == oi.fused
        vec  == [X \in Tens, g \in Grps |-> RadixVec(fused[g], X, sz[g])]
        vecU == [X \in Tens, g \in Grps |-> RadixVec(unf[g], X, sz[g])]
        dec  == [g \in Grps |-> Decode(ord[g], ext, sz[g])]
        offA(lam, i, h) == d.A.base + vec["A", "L"][lam] + vec["A", "I"][i] + vec["A", "K"][h]
        offB(lam, h, j) == d.B.base + vec["B", "L"][lam] + vec["B", "K"][h] + vec["B", "J"][j]
        offC(lam, i, j) == d.C.base + vec["C", "L"][lam] + vec["C", "I"][i] + vec["C", "J"][j]
        \* K2/K3 gather into the canonical buffers (one modulus plane).
        bufA == [q \in 0..(b * m * kh - 1) |->
                   LET lam == q \div (m * kh)  i == (q \div kh) % m  h == q % kh
                   IN  IF h < k THEN Cell(offA(lam, i, h)) ELSE Pad]
        bufB == [q \in 0..(b * nh * kh - 1) |->
                   LET lam == q \div (nh * kh)  j == (q \div kh) % nh  h == q % kh
                   IN  IF h < k /\ j < n THEN Cell(offB(lam, h, j)) ELSE Pad]
        \* D1: a packed post-swap role reads the caller's pack instead (§6b.4):
        \* role A at (λ·m + i)·k̂ + h; role B rows j < n̂ at batch stride
        \* r·k̂ = n·k̂ (Drop "PackBStrideNh": v1's n̂·k̂), rows j >= n being
        \* the next block's or the tail.
        pkA == "A" \in oi.proles
        pkB == "B" \in oi.proles
        pg  == [R \in oi.proles |-> PkInfo(od, oi.pR[R]).buf]
        bstr == IF Drop = "PackBStrideNh" THEN nh ELSE n
        qa(t) == (t[1] * m + t[2]) * kh + t[4]
        qb(t) == (t[1] * bstr + t[3]) * kh + t[4]
        ca(t) == IF pkA THEN PRead(pg["A"], qa(t)) ELSE bufA[(t[1] * m + t[2]) * kh + t[4]]
        cb(t) == IF pkB THEN PRead(pg["B"], qb(t)) ELSE bufB[(t[1] * nh + t[3]) * kh + t[4]]
        kind == IF b * m * n = 0 THEN "NOOP" ELSE IF k = 0 THEN "SCALE"
                ELSE IF mode = "DIRECT" \/ grp["K"] = {} THEN "DIRECT" ELSE "GEMM"
        \* K4 over the padded box; K5 scatters columns j < n only.
        gdom == {<<lam, i, j, h>> : lam \in 0..(b - 1), i \in 0..(m - 1),
                                     j \in 0..(nh - 1), h \in 0..(kh - 1)}
        \* D1: with a packed B, the products in columns j >= n read real
        \* elements of other rows (finite int8, §6b.4) and land in W columns
        \* K5 never reads: they are not contributions. (Unpacked: the pad rows
        \* are zero, and a nonzero product there is caught as Unscattered.)
        gdomC == IF pkB THEN {t \in gdom : t[3] < n} ELSE gdom
        nz   == {t \in gdomC : ~ca(t)[1] /\ ~cb(t)[1]}
        cadr(t) == IF t[3] < n THEN offC(t[1], t[2], t[3]) ELSE Unscattered
        \* DIRECT: native sum over the canonical k-bar order, no padding.
        ddom == {<<lam, i, j, h>> : lam \in 0..(b - 1), i \in 0..(m - 1),
                                     j \in 0..(n - 1), h \in 0..(k - 1)}
        contrib == CASE kind = "GEMM"
                          -> MkBag(nz, [t \in nz |-> <<cadr(t), ca(t)[2], cb(t)[2]>>])
                     [] kind = "DIRECT"
                          -> MkBag(ddom, [t \in ddom |-> <<offC(t[1], t[2], t[3]),
                                                           offA(t[1], t[2], t[4]),
                                                           offB(t[1], t[4], t[3])>>])
                     [] OTHER -> EmptyBag
        wdom == {<<lam, i, j>> : lam \in 0..(b - 1), i \in 0..(m - 1), j \in 0..(n - 1)}
        writes == IF kind = "NOOP" THEN EmptyBag
                  ELSE MkBag(wdom, [w \in wdom |-> offC(w[1], w[2], w[3])])
        \* the elements a GEMM reads: a packed operand's are those imd_pack read
        cells(bf) == {bf[q][2] : q \in {q \in DOMAIN bf : ~bf[q][1]}}
        readsA == CASE kind = "GEMM"   -> IF pkA THEN cells(pg["A"]) ELSE cells(bufA)
                    [] kind = "DIRECT" -> {offA(t[1], t[2], t[4]) : t \in ddom}
                    [] OTHER           -> {}
        readsB == CASE kind = "GEMM"   -> IF pkB THEN cells(pg["B"]) ELSE cells(bufB)
                    [] kind = "DIRECT" -> {offB(t[1], t[4], t[3]) : t \in ddom}
                    [] OTHER           -> {}
        \* D1 pack cells K4 reads (every row j < n̂ in role B)
        pread == [R \in oi.proles |-> IF R = "A" THEN {qa(t) : t \in gdom} ELSE {qb(t) : t \in gdom}]
        \* D1 KEY SOUNDNESS's right-hand side: the pack's image equals the
        \* plan's own operand buffer for X (design strides: role B rows j < n
        \* at batch stride r·k̂ = n·k̂).
        gatherEq == [R \in oi.proles |->
                       IF R = "A"
                       THEN \A q \in DOMAIN bufA : PRead(pg["A"], q) = bufA[q]
                       ELSE \A lam \in 0..(b - 1), j \in 0..(n - 1), h \in 0..(kh - 1) :
                              PRead(pg["B"], (lam * n + j) * kh + h) = bufB[(lam * nh + j) * kh + h]]
    IN  [d |-> d, swap |-> sw, labs |-> labs, ext |-> ext, kept |-> oi.kept, cls |-> oi.cls, grp |-> grp,
         sz |-> sz, m |-> m, n |-> n, k |-> k, b |-> b, kh |-> kh, nh |-> nh,
         ord |-> ord, unf |-> unf, fused |-> fused, vec |-> vec, vecU |-> vecU,
         dec |-> dec, bufA |-> bufA, bufB |-> bufB, kind |-> kind,
         contrib |-> contrib, writes |-> writes, readsA |-> readsA, readsB |-> readsB,
         ord15 |-> oi.ord15, proles |-> oi.proles, adopted |-> oi.adopted, pR |-> oi.pR,
         pg |-> pg, pread |-> pread, gatherEq |-> gatherEq, keyEq |-> oi.keyEq,
         slotKey |-> oi.slotKey, packKey |-> oi.packKey, partnerOnly |-> oi.partnerOnly,
         orderAgree |-> oi.orderAgree, mis |-> oi.mis]

-----------------------------------------------------------------------------
(* Einsum reference (DESIGN §1).                                          *)

Einsum(d) ==
    LET labs == AllLab(d)
        ext  == [l \in labs |-> Ext(d, l)]
        pts  == {x \in [labs -> 0..MaxIdx(d)] : \A l \in labs : x[l] < ext[l]}
        addr(X, x) == d[X].base + SumOver(Pos(d, X), [p \in Pos(d, X) |-> d[X].s[p] * x[d[X].l[p]]])
        clabs == Lab(d, "C")
        cpts == {y \in [clabs -> 0..MaxIdx(d)] : \A l \in clabs : y[l] < ext[l]}
    IN  [bag    |-> MkBag(pts, [x \in pts |-> <<addr("C", x), addr("A", x), addr("B", x)>>]),
         reads  |-> [X \in {"A", "B"} |-> {addr(X, x) : x \in pts}],
         \* C output positions (assignments of C's labels): one write each
         cpos   |-> MkBag(cpts, [y \in cpts |-> addr("C", y)]),
         pts    |-> pts]

-----------------------------------------------------------------------------
(* GEMM dimensions, the coalescing role swap, dispatch and §4a sizing.     *)

\* m, n, k, b of the GEMM view of d in d's OWN roles (no swap): a closed
\* form over the classification (what imd_workspace_size can compute
\* without building the lowering). kempty: no K label survives step 4.
UDims(d) ==
    LET labs == AllLab(d)
        ext  == [l \in labs |-> Ext(d, l)]
        kept == {l \in labs : ext[l] # 1}
        grp(g) == {l \in kept : ClassOf(d, l) = g}
    IN  [m |-> ProdOver(grp("I"), ext), n |-> ProdOver(grp("J"), ext),
         k |-> ProdOver(grp("K"), ext), b |-> ProdOver(grp("L"), ext),
         kempty |-> grp("K") = {}]

\* Empty output (b*m*n = 0); invariant under the swap.
EmptyOut(d) == LET u == UDims(d) IN u.b * u.m * u.n = 0

\* §5 step 8 coalescing. The candidate modes are C's modes with extent > 1
\* (Drop "SwapFilter": all of C's modes); FastMode is the one of smallest
\* |stride| (first in mode order on a tie: SwapWellDefined shows there is
\* no tie whenever the stride rule holds). The roles are swapped iff its
\* label is an I label (membership exactly {A, C}).
SwapCand(d)  == IF Drop = "SwapFilter" THEN Pos(d, "C")
                ELSE {p \in Pos(d, "C") : d.C.e[p] > 1}
FastMode(d)  == CHOOSE p \in SwapCand(d) :
                  \A q \in SwapCand(d) : Abs(d.C.s[p]) < Abs(d.C.s[q])
                                         \/ (Abs(d.C.s[p]) = Abs(d.C.s[q]) /\ p <= q)
Member(d, l) == {X \in Tens : l \in Lab(d, X)}
DoSwap(d)    == /\ Drop # "NoSwap"
                /\ SwapCand(d) # {}
                /\ Member(d, d.C.l[FastMode(d)]) = {"A", "C"}
\* Cᵀ = BᵀAᵀ: the operand descriptors exchange roles (so I <-> J).
Sw(d)        == [A |-> d.B, B |-> d.A, C |-> d.C]
\* The descriptor the lowering, the sizing and the limits work on.
LD(d)        == IF DoSwap(d) THEN Sw(d) ELSE d
PDims(d)     == UDims(LD(d))                         \* post-swap (DESIGN)
SDims(d)     == IF Drop = "SwapLate" THEN UDims(d) ELSE PDims(d)  \* what sizing uses

LowerTopP(d, mode, p) == LowerP(LD(d), mode, DoSwap(d), d, p)
LowerTop(d, mode)     == LowerTopP(d, mode, NoPk)
OrdInfoTop(d, p)      == OrdInfo(LD(d), DoSwap(d), d, p)
\* D1: PACK_MISMATCH of a plan of d created with packs p.
PackMis(d, p) == p # NoPk /\ OrdInfoTop(d, p).mis
\* The post-swap roles that are packed.
PRolesOf(d, p) == {IF DoSwap(d) THEN (IF op = "A" THEN "B" ELSE "A") ELSE op :
                     op \in {o \in {"A", "B"} : IsPk(p[o])}}

\* Dispatch by descriptor (§4 DIRECT path, §4a): a pure function of the
\* GEMM dimensions and the mode.
DispKind(u, mode) ==
    IF u.b * u.m * u.n = 0 THEN "NOOP"
    ELSE IF u.k = 0 THEN "SCALE"
    ELSE IF mode = "DIRECT" \/ u.kempty THEN "DIRECT"
    ELSE "GEMM"

\* §4a regions (one modulus plane, N = g = 1) as element counts, and their
\* element widths in bytes.
RegNames == {"sA", "sB", "fA", "fB", "vec", "RA", "RB", "W"}
RegCells(u, r) ==
    CASE r \in {"sA", "fA"} -> u.b * u.m
      [] r \in {"sB", "fB"} -> u.b * u.n
      [] r = "vec"          -> 2 * u.m + 2 * u.n + 2 * u.k + 3 * u.b
      [] r = "RA"           -> u.b * u.m * CeilQ(u.k)
      [] r = "RB"           -> u.b * CeilQ(u.n) * CeilQ(u.k)
      [] r = "W"            -> u.b * u.m * CeilQ(u.n)
RegWidth(r) == CASE r \in {"sA", "sB"} -> 2 [] r = "vec" -> 8 [] OTHER -> 1
\* §4a: 0 for an empty output; only vec for DIRECT, K-empty and k = 0.
\* D1 (§6b.6): sX, fX, RX of a packed post-swap role X ∈ pr have 0 bytes.
PackedRegs(pr) == UNION {IF X = "A" THEN {"sA", "fA", "RA"} ELSE {"sB", "fB", "RB"} : X \in pr}
SizedCellsP(u, mode, pr) ==
    LET kd == DispKind(u, mode) IN
    [r \in RegNames |-> CASE kd = "NOOP" -> 0
                          [] kd = "GEMM" -> IF r \in PackedRegs(pr) THEN 0 ELSE RegCells(u, r)
                          [] OTHER       -> IF r = "vec" THEN RegCells(u, r) ELSE 0]
SizedCells(u, mode) == SizedCellsP(u, mode, {})
AlignUp(x) == ((x + AlignWS - 1) \div AlignWS) * AlignWS
WsSizeU(u, mode) ==
    LET c == SizedCells(u, mode) IN
    SumOver(RegNames, [r \in RegNames |-> AlignUp(c[r] * RegWidth(r))])
WsSizeImpl(d, mode)   == WsSizeU(SDims(d), mode)   \* imd_workspace_size as built
WsSizeDesign(d, mode) == WsSizeU(PDims(d), mode)   \* DESIGN §4a

\* §5 row 5, the CUTLASS GemmCoord limit (scaled: GemmMax). It applies
\* exactly when the emulated GEMM runs: EMULATE mode, a non-empty output
\* and k >= 2 (k = 0: C <- beta*C; k = 1: K-empty, dispatched to DIRECT),
\* the same condition as the RN acceptance rule (orchestrator, M1).
LimitApplies(d, x) == /\ x.mode = "EMULATE" \/ Drop = "GemmLimitDirect"
                      /\ ~EmptyOut(d) \/ Drop = "GemmLimitEmpty"
                      /\ UDims(d).k >= 2 \/ Drop = "GemmLimitK"
LimitExceeded(d)   == LET u == SDims(d) IN
                      \/ u.m > GemmMax
                      \/ (IF Drop = "GemmLimitN" THEN u.n ELSE CeilQ(u.n)) > GemmMax

\* §3.2 "RN acceptance", accuracy policy: plan time rejects kappa >= T,
\* kappa = ceil(sqrt(k)) exact (NOT_SUPPORTED, "N too small for this k"),
\* only in EMULATE with a non-empty output and k >= 2 (the emulated GEMM
\* runs), only for rounding RN, independent of alpha. T = T(N) comes with
\* the call's options (x.tn); x.rn = rounding RN. As built (src/api.cpp
\* imd_plan_create) it is checked after every §5 plan-time row, so it is
\* the last plan-time status. imd_workspace_size never applies it (§6).
CeilSqrt(x)  == CHOOSE c \in 0..x : c * c >= x /\ (c = 0 \/ (c - 1) * (c - 1) < x)
KappaApplies(d, x) == /\ x.mode = "EMULATE" \/ Drop = "KappaDirect"
                      /\ ~EmptyOut(d) \/ Drop = "KappaEmpty"
                      /\ PDims(d).k >= 2 \/ Drop = "KappaK"
                      /\ x.rn \/ Drop = "KappaTrunc"
KappaTooBig(d, x)  == CeilSqrt(PDims(d).k) >= x.tn

-----------------------------------------------------------------------------
(* D1: the packs a model enumerates (imd_pack_create's valid outputs).    *)

\* X's labels after §5 steps 2 and 4 (diagonals merged, extent 1 dropped).
PkLab(d, op) == {l \in Lab(d, op) : Ext(d, l) # 1}
\* The classes of X as the plan sees them (what imd_pack_desc_for fills):
\* K = X's labels not in C, L = in C and in the partner, R = the rest.
PlanCls(d, op) ==
    LET pt == IF op = "A" THEN "B" ELSE "A" IN
    [K |-> {l \in PkLab(d, op) : l \notin Lab(d, "C")},
     L |-> {l \in PkLab(d, op) : l \in Lab(d, "C") /\ l \in Lab(d, pt)},
     R |-> {l \in PkLab(d, op) : l \in Lab(d, "C") /\ l \notin Lab(d, pt)}]
PermSeqs(S) == {f \in [1..Cardinality(S) -> S] : \A i, j \in DOMAIN f : i # j => f[i] # f[j]}
ClassChoices(d, op) ==
    IF PackEnum \in {"classes", "all"}
    THEN LET S == PkLab(d, op) IN
         UNION {UNION {{[K |-> pK, L |-> pL, R |-> pR] :
                           pK \in PermSeqs(SK), pL \in PermSeqs(SL),
                           pR \in PermSeqs(S \ (SK \cup SL))} : SL \in SUBSET (S \ SK)}
                : SK \in SUBSET S}
    ELSE LET c == PlanCls(d, op) IN
         {[K |-> pK, L |-> pL, R |-> pR] :
             pK \in PermSeqs(c.K), pL \in PermSeqs(c.L), pR \in PermSeqs(c.R)}
PackStrides(d, op) ==
    IF PackEnum \in {"strides", "all"}
    THEN (IF op = "A" THEN StridesA(Rk(d, "A")) ELSE StridesB(Rk(d, "B")))
    ELSE {d[op].s}
\* imd_pack_layout's NOT_SUPPORTED (§6b.3): K empty after lowering (k = 1),
\* and κ >= T with RN (the pack's N and rounding are the call's: x.tn, x.rn).
PackValid(d, x, pck) ==
    LET k == ProdOver(SeqSet(pck.K), [l \in Lab(d, pck.op) |-> Ext(d, l)]) IN
    /\ pck.K # <<>>
    /\ ~(x.rn /\ k >= 2 /\ CeilSqrt(k) >= x.tn)
PacksFor(d, x, op) ==
    {pck \in {[op |-> op, s |-> s, K |-> c.K, L |-> c.L, R |-> c.R] :
                 s \in PackStrides(d, op), c \in ClassChoices(d, op)} : PackValid(d, x, pck)}
Packings(d, x) ==
    CASE Packing = "none" -> {NoPk}
      [] Packing = "A"    -> {[A |-> p, B |-> NoPack] : p \in PacksFor(d, x, "A")}
      [] Packing = "B"    -> {[A |-> NoPack, B |-> p] : p \in PacksFor(d, x, "B")}
      [] Packing = "AB"   -> {[A |-> p, B |-> q] : p \in PacksFor(d, x, "A"), q \in PacksFor(d, x, "B")}

-----------------------------------------------------------------------------
(* Execute-time arguments (imd_contract) and checks (DESIGN §5 rows 6-7,  *)
(* §6). Addresses are in BYTES: element a of tensor X occupies            *)
(* [8a + mis[X], 8a + mis[X] + 8), mis[X] in 0..7 = misalignment of X's   *)
(* pointer. An Execs record is                                             *)
(*   [mode |-> "EMULATE" | "DIRECT",                                       *)
(*    nul  |-> subset of {"A", "B", "C", "ws"}: null pointers,             *)
(*    a0   |-> alpha = 0 (only the §6 null rule depends on it here),       *)
(*    rn   |-> rounding = RN (the §3.2 kappa policy applies to RN only),   *)
(*    tn   |-> T = T(N) of the chosen N (real: T(2) = 180 and up; the      *)
(*             models use a small symbolic T so kappa >= T is reachable),  *)
(*    mis  |-> [Tens -> 0..7],                                             *)
(*    wsp  |-> <<anchor, side, delta>>: where ws is placed. anchor "far":  *)
(*             FarWs; else ws = (byte hull end of anchor) + delta (side    *)
(*             "after") or (hull start) - workspace_size + delta ("before")*)
(*             so delta = 0 TOUCHES the anchor and +-1 overlaps one byte,  *)
(*    dws  |-> ws_bytes - workspace_size].                                 *)

XDef  == [mode |-> "EMULATE", nul |-> {}, a0 |-> FALSE, rn |-> TRUE, tn |-> 180,
          mis |-> [X \in Tens |-> 0], wsp |-> <<"far", "after", 0>>, dws |-> 0]
FarWs == 1024 * AlignWS

\* Byte hull of X: [8*lo + mis, 8*hi + mis) from the element interval.
BHull(d, x, X) ==
    LET iv == Interval(d, X) IN
    IF iv[1] = iv[2] THEN <<0, 0>>
    ELSE <<8 * iv[1] + x.mis[X], 8 * iv[2] + x.mis[X]>>
WsAddr(d, x) ==
    LET a == x.wsp[1] IN
    IF a = "far" THEN FarWs
    ELSE IF x.wsp[2] = "after" THEN BHull(d, x, a)[2] + x.wsp[3]
    ELSE BHull(d, x, a)[1] - WsSizeDesign(d, x.mode) + x.wsp[3]
WsBytesArg(d, x) == Max(0, WsSizeDesign(d, x.mode) + x.dws)       \* ws_bytes (size_t)
\* [ws, ws + workspace_size): the bytes the call actually uses.
WsRange(d, x)    == <<WsAddr(d, x), WsAddr(d, x) + WsSizeImpl(d, x.mode)>>
WsRangeD(d, x)   == <<WsAddr(d, x), WsAddr(d, x) + WsSizeDesign(d, x.mode)>>

CAliasB(d, x) == \/ Intersect(BHull(d, x, "C"), BHull(d, x, "A"))
                 \/ Intersect(BHull(d, x, "C"), BHull(d, x, "B"))
WsHitR(d, x, r) == \E X \in Tens : Intersect(r, BHull(d, x, X))
\* §6 1 (DESIGN v1.1 after M1): C must always be non-null; A and B only
\* when alpha # 0 and k # 0 (otherwise they are never read); ws only when
\* workspace_size > 0. sz is the workspace_size used for the test.
NullBad(d, x, sz) ==
    \/ "C" \in x.nul
    \/ {"A", "B"} \cap x.nul # {} /\ ~x.a0 /\ UDims(d).k # 0
    \/ "ws" \in x.nul /\ sz > 0
MisAl(d, x)   == (\E X \in Tens : x.mis[X] # 0) \/ WsAddr(d, x) % AlignWS # 0

-----------------------------------------------------------------------------
(* Status: the implementation chain. Plan time (imd_plan_create, §5 rows  *)
(* 1-5) and then execute time (imd_contract, §6 order). `off` = rules      *)
(* switched off (normally {Drop}).                                         *)

\* §5 rows 1-5 (plan time, before any D1 or §3.2 check).
BaseStatus(d, x, off) ==
    LET on(r) == r \notin off IN
          IF on("Extents") /\ ~ExtentsConsistent(d)                THEN "INVALID_VALUE"
          ELSE IF on("RepeatC") /\ RepeatedC(d)                     THEN "NOT_SUPPORTED"
          ELSE IF on("COnly") /\ COnlyBig(d)                        THEN "NOT_SUPPORTED"
          ELSE IF on("CStride") /\ ~Empty(d, "C") /\ ~StrideRule(d, "C")
                                                                    THEN "INVALID_VALUE"
          ELSE IF on("GemmLimit") /\ LimitApplies(d, x) /\ LimitExceeded(d)
                                                                    THEN "INVALID_VALUE"
          ELSE "SUCCESS"

\* p: the packs (D1 imd_plan_create_packed; NoPk = imd_plan_create). After
\* §5 rows 1-5: PACK_MISMATCH (the key of each packed slot, §6b.5), then
\* the §3.2 RN acceptance LAST (Drop "PackAfterKappa": κ before the key).
StatusXP(d, x, p, off) ==
    LET on(r) == r \notin off
        base == BaseStatus(d, x, off)
        pm   == on("PackCheck") /\ PackMis(d, p)
        kap  == on("Kappa") /\ KappaApplies(d, x) /\ KappaTooBig(d, x)
        plan ==
          IF base # "SUCCESS" THEN base
          ELSE IF Drop = "PackAfterKappa"
               THEN (IF kap THEN "NOT_SUPPORTED" ELSE IF pm THEN "PACK_MISMATCH" ELSE "SUCCESS")
          ELSE IF pm  THEN "PACK_MISMATCH"
          ELSE IF kap THEN "NOT_SUPPORTED"
          ELSE "SUCCESS"
        wsr == IF Drop = "WsRangeBytes" THEN <<WsAddr(d, x), WsAddr(d, x) + WsBytesArg(d, x)>>
               ELSE WsRange(d, x)
        cNull  == <<on("Null") /\ CASE Drop = "NullStrict" -> x.nul # {}
                                     [] Drop = "NullWsLax"  ->
                                          NullBad(d, [x EXCEPT !.nul = @ \ {"ws"}], 0)
                                     [] OTHER -> NullBad(d, x, WsSizeImpl(d, x.mode)),
                    "INVALID_VALUE">>
        cEmpty == <<EmptyOut(d), "SUCCESS">>
        cAlias == <<(on("Aliasing") /\ CAliasB(d, x)) \/ (on("WsAlias") /\ WsHitR(d, x, wsr)),
                    "ALIASING">>
        cMis   == <<on("Misaligned") /\ MisAl(d, x), "MISALIGNED">>
        cIns   == <<on("Insufficient") /\ WsBytesArg(d, x) < WsSizeImpl(d, x.mode),
                    "INSUFFICIENT_WORKSPACE">>
        chk == CASE Drop = "OrderNullEmpty"   -> <<cEmpty, cNull, cAlias, cMis, cIns>>
                 [] Drop = "OrderEmptyLast"   -> <<cNull, cAlias, cMis, cIns, cEmpty>>
                 [] Drop = "OrderAlignFirst"  -> <<cNull, cEmpty, cMis, cAlias, cIns>>
                 [] Drop = "OrderInsuffFirst" -> <<cNull, cEmpty, cAlias, cIns, cMis>>
                 [] OTHER                     -> <<cNull, cEmpty, cAlias, cMis, cIns>>
        fire == {t \in 1..5 : chk[t][1]}
    IN  IF plan # "SUCCESS" THEN plan
        ELSE IF fire = {} THEN "SUCCESS"
        ELSE chk[CHOOSE t \in fire : \A u \in fire : t <= u][2]

StatusX(d, x, off)   == StatusXP(d, x, NoPk, off)
FullStatusP(d, x, p) == StatusXP(d, x, p, {Drop})
FullStatus(d, x)     == StatusX(d, x, {Drop})
\* The status of a descriptor under the default execute arguments XDef
\* (EMULATE, no null, aligned, workspace far away and large enough).
Status(d) == FullStatus(d, XDef)

-----------------------------------------------------------------------------
(* Enumeration *)

\* Restricted-growth labelling: label t+1 first appears after label t.
Canon(sq) == \A p \in 1..Len(sq) : sq[p] <= 1 + SetMax({sq[q] : q \in 1..(p - 1)} \cup {0})

ExtChoices(lA, lB, lC) ==
    IF PerModeExtents
    THEN {<<eA, eB, eC>> : eA \in [1..Len(lA) -> ExtentSet],
                           eB \in [1..Len(lB) -> ExtentSet],
                           eC \in [1..Len(lC) -> ExtentSet]}
    ELSE LET used == {lA[p] : p \in 1..Len(lA)} \cup {lB[p] : p \in 1..Len(lB)}
                       \cup {lC[p] : p \in 1..Len(lC)}
         IN  {<<[p \in 1..Len(lA) |-> x[lA[p]]], [p \in 1..Len(lB) |-> x[lB[p]]],
                [p \in 1..Len(lC) |-> x[lC[p]]]>> : x \in [used -> ExtentSet]}

Tensor(bs, l, e, s) == [base |-> bs, l |-> l, e |-> e, s |-> s]

Init ==
    /\ \E rA \in 0..RankA, rB \in 0..RankB, rC \in 0..RankC :
       \E lA \in [1..rA -> Labels], lB \in [1..rB -> Labels], lC \in [1..rC -> Labels] :
         /\ ~CanonicalLabels \/ Canon(lA \o lB \o lC)
         /\ \E ee \in ExtChoices(lA, lB, lC),
               sA \in StridesA(rA), sB \in StridesB(rB), sC \in StridesC(rC),
               bA \in BasesA, bB \in BasesB, bC \in BasesC :
              desc = [A |-> Tensor(bA, lA, ee[1], sA),
                      B |-> Tensor(bB, lB, ee[2], sB),
                      C |-> Tensor(bC, lC, ee[3], sC)]
    /\ xa \in Execs
    \* D1: every valid pack of the model's space (one, arbitrary, when the
    \* descriptor already fails §5 rows 1-5: the packs cannot matter there).
    /\ pk \in LET P == Packings(desc, xa) IN
              IF Packing = "none" \/ BaseStatus(desc, xa, {Drop}) = "SUCCESS" \/ P = {} THEN P
              ELSE {CHOOSE p \in P : TRUE}
    /\ st  = FullStatusP(desc, xa, pk)
    \* (D1: the lowering is also built for a plan that passes rows 1-5 but is
    \* rejected by the key or κ, so KEY SOUNDNESS is checked on every pack.)
    /\ low = IF st = "SUCCESS" \/ (pk # NoPk /\ BaseStatus(desc, xa, {Drop}) = "SUCCESS")
             THEN LowerTopP(desc, xa.mode, pk) ELSE "none"
    /\ ein = IF st = "SUCCESS" /\ ExtentsConsistent(desc) THEN Einsum(desc) ELSE "none"
    /\ el  = [X \in Tens |-> Elems(desc, X)]

Next == UNCHANGED vars
Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Invariants. "ok" = the call was accepted (SUCCESS). The lowering works *)
(* on low.d, the descriptor in GEMM roles (= desc unless the roles were    *)
(* swapped); RoleA / RoleB name the USER tensor in GEMM role A / B.       *)

ok == st = "SUCCESS"

RoleA == IF low.swap THEN "B" ELSE "A"
RoleB == IF low.swap THEN "A" ELSE "B"
\* The einsum bag in GEMM roles: with Cᵀ = BᵀAᵀ every user contribution
\* C[c] += A[a]*B[b] is computed as C[c] += A'[b]*B'[a] (a product, so the
\* value is the same; the native path is fma(a_h, b_h, acc), symmetric too).
SwapTriples(bg) == [t \in {<<u[1], u[3], u[2]>> : u \in DOMAIN bg} |-> bg[<<t[1], t[3], t[2]>>]]
EinBag == IF low.swap THEN SwapTriples(ein.bag) ELSE ein.bag

\* DESIGN §5 "classification is a partition": the kept labels (extent # 1)
\* are split into I, J, K, L, each with the right membership set (in GEMM
\* roles: after a swap, the user's {B, C} labels are I).
Partition ==
    ok => /\ UNION {low.grp[g] : g \in Grps} = low.kept
          /\ \A g, h \in Grps : g # h => low.grp[g] \cap low.grp[h] = {}
          /\ \A l \in low.kept :
               LET M == {X \in Tens : l \in Lab(low.d, X)} IN
               CASE low.cls[l] = "I" -> M = {"A", "C"}
                                          \/ (M = {"C"} /\ (low.ext[l] = 0 \/ Drop = "COnly"))
                 [] low.cls[l] = "J" -> M = {"B", "C"}
                 [] low.cls[l] = "K" -> M \in {{"A", "B"}, {"A"}, {"B"}}
                 [] low.cls[l] = "L" -> M = Tens
                 [] OTHER            -> FALSE

\* DESIGN §5 "lowering preserves the einsum" and "Scatter o GEMM o Gather =
\* einsum": the packed, padded, fused, reordered (and possibly role-
\* swapped) GEMM view, or the DIRECT sum, produces exactly the einsum's
\* multiset of (addrC, addrA, addrB) contributions. A nonzero product in an
\* unscattered column would appear with addrC = Unscattered.
EinsumPreserved == ok => low.contrib = EinBag

\* DESIGN §5 "sorted-stride rule => scatter is injective", §4 "every C
\* element is written exactly once": no C address is written twice.
ScatterExactlyOnce == ok => \A a \in DOMAIN low.writes : low.writes[a] = 1

\* DESIGN §1 / §3.7: C <- alpha*sum + beta*C covers EVERY element of C (C's
\* own descriptor), once per C output position; nothing else is written.
ScatterCoversC ==
    ok => /\ DOMAIN low.writes = el["C"]
          /\ low.writes = ein.cpos

\* Every access stays inside the tensor as described by ITS OWN modes.
AccessInBounds ==
    ok => /\ low.readsA \subseteq el[RoleA]
          /\ low.readsB \subseteq el[RoleB]
          /\ DOMAIN low.writes \subseteq el["C"]

\* The gather (or the DIRECT kernel) reads exactly the elements the einsum
\* uses (no more, no less).
ReadsExact ==
    ok /\ low.kind \in {"GEMM", "DIRECT"} =>
        low.readsA = ein.reads[RoleA] /\ low.readsB = ein.reads[RoleB]

\* DESIGN §5 zero extents + §3.7 read rules: empty I/J/L -> no-op (nothing
\* read or written, and indeed C is empty and the einsum has no terms);
\* k = 0 -> C <- beta*C (all of C written once, A and B not read, no terms).
ZeroExtentRules ==
    ok => /\ (low.b * low.m * low.n = 0) = (el["C"] = {})
          /\ low.b * low.m * low.n = 0 =>
               /\ low.kind = "NOOP" /\ low.writes = EmptyBag
               /\ low.readsA = {} /\ low.readsB = {} /\ ein.bag = EmptyBag
          /\ low.b * low.m * low.n > 0 /\ low.k = 0 =>
               /\ low.kind = "SCALE" /\ ein.bag = EmptyBag
               /\ low.readsA = {} /\ low.readsB = {}
               /\ DOMAIN low.writes = el["C"]
          /\ (ein.bag = EmptyBag /\ el["C"] # {}) => low.k = 0

\* DESIGN §4/§4a padding: kh, nh are the PadQ-roundups; every K-pad cell
\* of A' and every K-pad cell or pad row (j >= n) of B' holds 0, and every
\* non-pad cell is a real gather. (That pad products reach no C element is
\* part of EinsumPreserved via the Unscattered sentinel.)
PaddingInert ==
    ok => /\ low.kh % PadQ = 0 /\ low.k <= low.kh /\ low.kh < low.k + PadQ
          /\ low.nh % PadQ = 0 /\ low.n <= low.nh /\ low.nh < low.n + PadQ
          /\ \A q \in DOMAIN low.bufA : low.bufA[q][1] <=> (q % low.kh >= low.k)
          /\ \A q \in DOMAIN low.bufB :
                low.bufB[q][1] <=> (q % low.kh >= low.k \/ (q \div low.kh) % low.nh >= low.n)

\* DESIGN §5 "gather is a bijection onto the canonical buffer" and §4a
\* region sizes: the encodings used by K2/K3/K4/K5 map their index boxes
\* one-to-one onto 0..size-1 of RA (b*m*kh), RB (b*nh*kh), W (b*m*nh).
LayoutBijective ==
    ok => LET m == low.m  n == low.n  b == low.b  kh == low.kh  nh == low.nh
              encA == {(lam * m + i) * kh + h : lam \in 0..(b - 1), i \in 0..(m - 1), h \in 0..(kh - 1)}
              encB == {(lam * nh + j) * kh + h : lam \in 0..(b - 1), j \in 0..(nh - 1), h \in 0..(kh - 1)}
              encW == {(lam * m + i) * nh + j : lam \in 0..(b - 1), i \in 0..(m - 1), j \in 0..(nh - 1)}
          IN  /\ encA = 0..(b * m * kh - 1)
              /\ encB = 0..(b * nh * kh - 1)
              /\ encW = 0..(b * m * nh - 1)

\* r_layout §4.2: canonical index <-> label assignment is a bijection per
\* group (first label fastest), AND it is the decode the lowering actually
\* uses: the (unfused) scatter vector at canonical index q is the einsum
\* address offset sum_l x[l]*s_X(l) of the label assignment x that q
\* decodes to. With FusionPreservesAddresses (vec = vecU) this ties the
\* scatter vectors on the lowering path to the bijection.
DecodeBijective ==
    ok => \A g \in Grps :
            LET G == low.grp[g]
                all == {x \in [G -> 0..MaxIdx(desc)] : \A l \in G : x[l] < low.ext[l]}
            IN  /\ {low.dec[g][q] : q \in DOMAIN low.dec[g]} = all
                /\ Cardinality(all) = low.sz[g]
                /\ \A X \in Tens, q \in DOMAIN low.dec[g] :
                      low.vecU[X, g][q] =
                        SumOver(G, [l \in G |-> low.dec[g][q][l] * MS(low.d, X, l)])

\* DESIGN §5 step 9: fusion preserves every address: the scatter vectors
\* built from the fused modes equal the label-by-label sums.
FusionPreservesAddresses ==
    ok => \A X \in Tens, g \in Grps : low.vec[X, g] = low.vecU[X, g]

\* DESIGN §5 step 10: off_X depends only on X's own groups (vA,J = 0,
\* vB,I = 0, vC,K = 0), so the three-term additive form is exact.
OperandIndependence ==
    ok => \A X \in Tens, g \in Grps :
            X \notin Car[g] => \A q \in DOMAIN low.vec[X, g] : low.vec[X, g][q] = 0

\* DESIGN §5 table / §4: C does not share an element with A or B.
NoCAliasing ==
    ok => el["C"] \cap (el["A"] \cup el["B"]) = {}

\* DESIGN §5 table: the sorted-|stride| rule is SUFFICIENT for injectivity.
\* Checked for all three tensors of every enumerated descriptor (accepted
\* or not), so A and B layouts add test cases for the lemma.
StrideRuleImpliesInjective ==
    \A X \in Tens : StrideRule(desc, X) => Injective(desc, X)

\* Expected to FAIL (negative model): the rule is not NECESSARY (DESIGN:
\* "it does reject some exotic non-overlapping layouts").
StrideRuleNecessary ==
    \A X \in Tens : (~Empty(desc, X) /\ Injective(desc, X)) => StrideRule(desc, X)

\* The DESIGN status table, written independently of the implementation
\* chain StatusX, in DESIGN's precedence order: §5 rows 1-5 (plan time),
\* then §6 imd_contract 1-5 (null, empty output, ALIASING = §5 row 6 with
\* its two halves, MISALIGNED = row 7, INSUFFICIENT_WORKSPACE).
\* <<rule name (the Drop that disables it), fires?, status>>. Every
\* condition here is the DESIGN one (post-swap m and nh, the byte range
\* [ws, ws + workspace_size)), so a model whose implementation deviates is
\* caught by StatusCorrect even when no semantic invariant notices.
\* DESIGN §3.2 / §6, written independently of KappaApplies: the policy
\* rejects iff the emulated GEMM would run with RN and ceil(sqrt(k)) >= T,
\* in closed form: ceil(sqrt(k)) >= T <=> (T - 1)^2 < k (for T >= 1).
KappaReject(d, x) ==
    LET k == PDims(d).k IN
    /\ x.mode = "EMULATE" /\ ~EmptyOut(d) /\ k >= 2 /\ x.rn
    /\ (x.tn - 1) * (x.tn - 1) < k
\* D1 (§6b.5): PACK_MISMATCH comes after §5 rows 1-5 and before the §3.2
\* RN acceptance. (Its condition is the key check of each packed slot; the
\* table checks its PLACE in the precedence.)
ReasonsP(d, x, p) ==
    LET u == PDims(d) IN
    << <<"Extents",   ~ExtentsConsistent(d),                     "INVALID_VALUE">>,
       <<"RepeatC",   RepeatedC(d),                              "NOT_SUPPORTED">>,
       <<"COnly",     COnlyBig(d),                               "NOT_SUPPORTED">>,
       <<"CStride",   ~Empty(d, "C") /\ ~StrideRule(d, "C"),     "INVALID_VALUE">>,
       <<"GemmLimit", x.mode = "EMULATE" /\ ~EmptyOut(d) /\ u.k >= 2
                      /\ (u.m > GemmMax \/ CeilQ(u.n) > GemmMax), "INVALID_VALUE">>,
       <<"PackCheck", PackMis(d, p),                             "PACK_MISMATCH">>,
       <<"Kappa",     KappaReject(d, x),                         "NOT_SUPPORTED">>,
       <<"Null",      NullBad(d, x, WsSizeDesign(d, x.mode)),    "INVALID_VALUE">>,
       <<"-",         EmptyOut(d),                               "SUCCESS">>,
       <<"Aliasing",  CAliasB(d, x),                             "ALIASING">>,
       <<"WsAlias",   WsHitR(d, x, WsRangeD(d, x)),              "ALIASING">>,
       <<"Misaligned", MisAl(d, x),                              "MISALIGNED">>,
       <<"Insufficient", WsBytesArg(d, x) < WsSizeDesign(d, x.mode),
                                                       "INSUFFICIENT_WORKSPACE">> >>
\* The status of the FIRST enabled reason that fires; SUCCESS if none.
ExpectedStatusP(d, x, p) ==
    LET R == ReasonsP(d, x, p)
        F == {t \in 1..Len(R) : R[t][1] # Drop /\ R[t][2]}
    IN  IF F = {} THEN "SUCCESS" ELSE R[CHOOSE t \in F : \A u \in F : t <= u][3]
ExpectedStatus(d, x) == ExpectedStatusP(d, x, NoPk)

\* DESIGN §5 / §6 statuses: each rejection has the right status and reason
\* and comes in DESIGN's precedence order (first firing reason wins, e.g.
\* C[i,i] with strides (1,1) is NOT_SUPPORTED, not INVALID_VALUE; a null
\* pointer beats an empty output; ALIASING beats MISALIGNED beats
\* INSUFFICIENT_WORKSPACE); acceptance of a non-empty output implies every
\* SEMANTIC legality fact (not just the cheap tests that imply them). A
\* aliasing B is never a reason to reject (no row of the table).
StatusCorrect ==
    /\ st \in {"SUCCESS", "INVALID_VALUE", "NOT_SUPPORTED", "ALIASING",
               "MISALIGNED", "INSUFFICIENT_WORKSPACE", "PACK_MISMATCH"}
    /\ st = ExpectedStatusP(desc, xa, pk)
    /\ st = "PACK_MISMATCH" => pk # NoPk
    /\ ~ExtentsConsistent(desc) => st = "INVALID_VALUE"
    /\ st = "NOT_SUPPORTED" => ExtentsConsistent(desc)
                               /\ (Unsupported(desc) \/ KappaReject(desc, xa))
    /\ ok => /\ ExtentsConsistent(desc)
             /\ ~Unsupported(desc)
             /\ Injective(desc, "C")
             /\ el["C"] \cap (el["A"] \cup el["B"]) = {}

\* DESIGN §5 step 8 (and §1.1: the k-bar order is a fixed function of the
\* descriptors): within I, J and L the canonical order is non-decreasing in
\* C's merged |stride| (so CRT scatter writes coalesce); within K it is
\* non-decreasing in the LARGER operand's |stride| (A if m >= n, else B, in
\* GEMM roles; stride 0 for a label the operand lacks). Ties are broken by a
\* fixed key (see CanonicalOrderExact). The "larger operand" is recomputed
\* here from the post-swap m, n. (D1: not for the classes whose order a
\* pack fixes, low.adopted; see AdoptedOrderExact.)
CanonicalOrderSorted ==
    ok => LET bigX == IF low.m >= low.n THEN "A" ELSE "B"
              keyT(g) == IF g = "K" THEN bigX ELSE "C"
          IN  \A g \in Grps \ low.adopted : \A t \in 1..(Len(low.ord[g]) - 1) :
                Abs(MS(low.d, keyT(g), low.ord[g][t]))
                  <= Abs(MS(low.d, keyT(g), low.ord[g][t + 1]))

\* DESIGN §5 step 8 in full (orchestrator clarification, as built in M1):
\* the swap comes FIRST; then I by (|s_C|, |s_A|, position), J by (|s_C|,
\* |s_B|, position), L by (|s_C|, |s_A|, |s_B|, position) and K by (|s_big|,
\* |s_other|, position), big = A if m >= n else B, all in the POST-swap
\* roles (low.d, low.m, low.n), and position = first occurrence in the
\* CALLER's A.labels ++ B.labels ++ C.labels (desc). Lexicographic keys;
\* every group's order is exactly the sorted one. Written independently of
\* Lower's key table.
CanonicalOrderExact ==
    ok => LET d == low.d
              bigX == IF low.m >= low.n THEN "A" ELSE "B"
              othX == IF bigX = "A" THEN "B" ELSE "A"
              cat == desc.A.l \o desc.B.l \o desc.C.l
              pos(l) == CHOOSE p \in 1..Len(cat) : cat[p] = l /\ \A q \in 1..(p - 1) : cat[q] # l
              s(X, l) == Abs(MS(d, X, l))
              kk(g, l) == CASE g = "I" -> <<s("C", l), s("A", l), pos(l)>>
                            [] g = "J" -> <<s("C", l), s("B", l), pos(l)>>
                            [] g = "L" -> <<s("C", l), s("A", l), s("B", l), pos(l)>>
                            [] g = "K" -> <<s(bigX, l), s(othX, l), pos(l)>>
              lt(u, v) == \E t \in DOMAIN u : u[t] < v[t] /\ \A r \in 1..(t - 1) : u[r] = v[r]
          IN  \A g \in Grps :
                /\ {low.ord[g][t] : t \in DOMAIN low.ord[g]} = low.grp[g]
                /\ g \notin low.adopted =>
                     \A t \in 1..(Len(low.ord[g]) - 1) :
                        lt(kk(g, low.ord[g][t]), kk(g, low.ord[g][t + 1]))

\* DESIGN §1.1 / r_layout §4.1: the result does not depend on the group
\* orders (canonical orders are a performance choice only). Every
\* permutation of every group, unfused, gives the einsum multiset.
Perms(S) == {f \in [1..Cardinality(S) -> S] : \A i, j \in DOMAIN f : i # j => f[i] # f[j]}
OrderIndependent ==
    (ok /\ CheckOrders) =>
      \A pI \in Perms(low.grp["I"]), pJ \in Perms(low.grp["J"]),
         pK \in Perms(low.grp["K"]), pL \in Perms(low.grp["L"]) :
        LET o == [I |-> pI, J |-> pJ, K |-> pK, L |-> pL]
            dd == low.d
            md(g) == [t \in 1..Len(o[g]) |->
                        [e |-> low.ext[o[g][t]],
                         s |-> [X \in Tens |-> MS(dd, X, o[g][t])]]]
            v == [X \in Tens, g \in Grps |-> RadixVec(md(g), X, low.sz[g])]
            dom == {<<lam, i, j, h>> : lam \in 0..(low.b - 1), i \in 0..(low.m - 1),
                                        j \in 0..(low.n - 1), h \in 0..(low.k - 1)}
            tr(t) == <<dd.C.base + v["C", "L"][t[1]] + v["C", "I"][t[2]] + v["C", "J"][t[3]],
                       dd.A.base + v["A", "L"][t[1]] + v["A", "I"][t[2]] + v["A", "K"][t[4]],
                       dd.B.base + v["B", "L"][t[1]] + v["B", "K"][t[4]] + v["B", "J"][t[3]]>>
        IN  MkBag(dom, [t \in dom |-> tr(t)]) = EinBag

-----------------------------------------------------------------------------
(* The coalescing role swap (DESIGN §5 step 8).                           *)

\* C's modes with extent > 1 and the smallest |stride| among them.
BigC(d) == {p \in Pos(d, "C") : d.C.e[p] > 1}
MinS(d) == LET S == {Abs(d.C.s[p]) : p \in BigC(d)} IN CHOOSE v \in S : \A w \in S : v <= w

\* (c) "The smallest |stride| is unique whenever rule 4 holds": over every
\* enumerated descriptor with a non-empty C that satisfies the sorted-
\* |stride| rule, exactly one mode of extent > 1 has the smallest |stride|,
\* so the swap decision does not depend on a tie-break.
SwapWellDefined ==
    (~Empty(desc, "C") /\ StrideRule(desc, "C") /\ BigC(desc) # {}) =>
        Cardinality({p \in BigC(desc) : Abs(desc.C.s[p]) = MinS(desc)}) = 1

\* (b) "so that j-bar is C's fastest axis": if the fastest C mode (extent
\* > 1) carries an I or a J label, then after the (possible) swap it is a J
\* label, it is the FIRST label of the canonical J order, and consecutive
\* j-bar indices are exactly that smallest |stride| apart in C's scatter
\* vector (fusion included). (If it carries an L label, nothing is claimed.)
\* D1: with a packed post-swap B the J order is the pack's (§6b.2
\* "Performance consequence": the coalescing goal is lost); the swap
\* itself is unchanged.
SwapMakesJFastest ==
    (ok /\ ~EmptyOut(desc) /\ BigC(desc) # {}) =>
        LET S  == {p \in BigC(desc) : Abs(desc.C.s[p]) = MinS(desc)}
            fl == desc.C.l[CHOOSE p \in S : TRUE]
        IN  Member(desc, fl) \in {{"A", "C"}, {"B", "C"}} =>
              /\ fl \in low.grp["J"]
              /\ "J" \notin low.adopted =>
                   /\ low.ord["J"][1] = fl
                   /\ Abs(low.vec["C", "J"][1] - low.vec["C", "J"][0]) = MinS(desc)

-----------------------------------------------------------------------------
(* Dispatch and §4a sizing.                                               *)

\* Elements of each §4a region the call actually uses: K1 writes one
\* sigma / flag per GEMM row of A' and column of B'; K0 copies the nine
\* scatter vectors the lowering built; K2/K3 fill the canonical buffers; K4
\* writes the b*m*nh plane. DIRECT and SCALE use only vec. D1: no K1, K2
\* or K3 runs for a packed post-swap role (it reads the caller's pack).
UsedCells ==
    LET kd == low.kind
        pairs == {p \in Tens \X Grps : p[1] \in Car[p[2]]}
        gemm(r, v) == IF kd = "GEMM" /\ r \notin PackedRegs(low.proles) THEN v ELSE 0
    IN  [r \in RegNames |->
           CASE kd = "NOOP"   -> 0
             [] r \in {"sA", "fA"} -> gemm(r, low.b * low.m)
             [] r \in {"sB", "fB"} -> gemm(r, low.b * low.n)
             [] r = "vec" -> SumOver(pairs, [p \in pairs |-> Cardinality(DOMAIN low.vec[p])])
             [] r = "RA"  -> gemm(r, Cardinality(DOMAIN low.bufA))
             [] r = "RB"  -> gemm(r, Cardinality(DOMAIN low.bufB))
             [] r = "W"   -> gemm(r, low.b * low.m * low.nh)]

\* §4 / §4a "K-empty -> DIRECT is documented dispatch by descriptor": the
\* path taken is the closed-form function DispKind of the (post-swap)
\* dimensions and the mode, which is also what sizes the workspace.
DispatchByDescriptor ==
    ok => /\ low.kind = DispKind(PDims(desc), xa.mode)
          /\ <<low.m, low.n, low.k, low.b>> =
               <<PDims(desc).m, PDims(desc).n, PDims(desc).k, PDims(desc).b>>

\* §4a "workspace_size" covers every element the call uses, region by
\* region (a region overrun corrupts the next region or runs past the end).
WorkspaceFits ==
    ok => \A r \in RegNames : UsedCells[r] <= SizedCellsP(SDims(desc), xa.mode, PRolesOf(desc, pk))[r]

\* §4a is EXACT: each region is exactly what is used (and so, with the
\* swap, §4a's m and n are the post-swap ones; D1 §6b.6: a packed role's
\* sX, fX, RX are 0, on the post-swap role).
WorkspaceExact ==
    ok => UsedCells = SizedCellsP(SDims(desc), xa.mode, PRolesOf(desc, pk))

\* §4 launch limits / §5 row 5: every CUTLASS GEMM that is launched has
\* m <= GemmMax and nh <= GemmMax (GemmCoord is int).
GemmDimsFit ==
    ok /\ low.kind = "GEMM" => low.m <= GemmMax /\ low.nh <= GemmMax

\* §5 row 5 scope: a call that runs no CUTLASS GEMM (DIRECT mode, an empty
\* output, k = 0, or K-empty with k = 1) is never rejected because of the
\* GEMM limits.
LimitScope ==
    (xa.mode = "DIRECT" \/ EmptyOut(desc) \/ PDims(desc).k < 2) =>
        st = StatusXP(desc, xa, pk, {Drop, "GemmLimit"})

\* §3.2 accuracy policy: every accepted call that runs the emulated GEMM
\* with RN has kappa = ceil(sqrt(k)) < T (so 2T - kappa > T and the RN
\* shift rule is far from its soundness edge kappa = 2T).
KappaAccepted ==
    ok /\ low.kind = "GEMM" /\ xa.rn => CeilSqrt(low.k) < xa.tn

\* §3.2 / §6 scope of the policy: a call that runs no emulated GEMM
\* (DIRECT mode, an empty output, k = 0, K-empty with k = 1) or uses
\* TRUNC is never rejected because of it, whatever k and T are. Each
\* disjunct is necessary (neg_kappa_direct / _empty / _k / _trunc). The
\* k < 2 disjunct matters only when T <= 1 (kappa(1) = 1): at every real
\* N, T >= 180, so there it is implied by kappa < T; the model uses T = 1
\* to show the clause is still the right one (no GEMM runs at k = 1).
KappaScope ==
    (xa.mode = "DIRECT" \/ EmptyOut(desc) \/ PDims(desc).k < 2 \/ ~xa.rn) =>
        st = StatusXP(desc, xa, pk, {Drop, "Kappa"})

-----------------------------------------------------------------------------
(* imd_contract execute-time checks (DESIGN §5 rows 6-7, §6).             *)

\* §6 1: no accepted call dereferences a null pointer: C whenever anything
\* is done, ws whenever the call uses workspace bytes, A and B when they
\* are read (GEMM / DIRECT with alpha # 0; alpha = 0 is C <- beta*C, DESIGN
\* §4, so A and B are not read, cf. Pipeline AlphaZeroNoABRead).
NoNullDeref ==
    ok /\ low.kind # "NOOP" =>
        /\ "C" \notin xa.nul
        /\ WsSizeImpl(desc, xa.mode) > 0 => "ws" \notin xa.nul
        /\ low.kind \in {"GEMM", "DIRECT"} /\ ~xa.a0 => {"A", "B"} \cap xa.nul = {}

\* §4 / §5 row 6: the workspace bytes the call uses, [ws, ws +
\* workspace_size), share no byte with any element of A, B or C (the GEMM
\* path writes the workspace while K2/K3/K5 read A, B and K5 writes C).
ElemBytesHit(x, X, a, r) == LET lo == 8 * a + x.mis[X] IN lo < r[2] /\ r[1] < lo + 8
WsNoAlias ==
    ok /\ low.kind # "NOOP" =>
        LET r == WsRangeD(desc, xa) IN
        \A X \in Tens : \A a \in el[X] : ~ElemBytesHit(xa, X, a, r)

\* §5 row 6 soundness: ALIASING is reported only when C's byte hull meets
\* A's or B's, or [ws, ws + workspace_size) meets a byte hull (the tests
\* are conservative hull tests, never "more" than that).
AliasingSound ==
    st = "ALIASING" => CAliasB(desc, xa) \/ WsHitR(desc, xa, WsRangeD(desc, xa))

\* §5 row 7: an accepted call that touches memory sees 8-byte aligned A,
\* B, C and an AlignWS-aligned workspace (so every region offset of §4a,
\* a multiple of AlignWS, is aligned in absolute terms).
AlignedAccess ==
    ok /\ low.kind # "NOOP" =>
        (\A X \in Tens : xa.mis[X] = 0) /\ WsAddr(desc, xa) % AlignWS = 0

\* §6 5: an accepted call has ws_bytes >= workspace_size (DESIGN §4a).
WsSufficient ==
    ok => WsBytesArg(desc, xa) >= WsSizeDesign(desc, xa.mode)

-----------------------------------------------------------------------------
(* D1 packed operands (DESIGN §6b.2, §6b.5): the adopted lowering and the  *)
(* compatibility key. Every claim above also runs on the packed lowering   *)
(* (EinsumPreserved with the packed operand read FROM THE PACK, for every  *)
(* order within a class the model enumerates).                            *)

\* A packed plan that passed §5 rows 1-5 (its lowering was built).
LowP == pk # NoPk /\ BaseStatus(desc, xa, {Drop}) = "SUCCESS"

\* KEY SOUNDNESS: equal keys => the pack's image (X's addresses at (λ, ρ,
\* k̄) in the pack's enumeration, §6b.4 layout, K padding zero) equals the
\* plan's own operand buffer for X: RA's rows for post-swap A; for post-swap
\* B the rows j < n, the pack read at batch stride r·k̂.
PackKeySound == LowP => \A R \in low.proles : low.keyEq[R] => low.gatherEq[R]

\* COMPLETENESS for the packs built to the plan: classes = the plan's
\* (imd_pack_desc_for's: K = X's labels not in C, L = in C and the
\* partner, R = the rest), any order within each class, X's own strides,
\* no partner-only K label, and (both packed) equal K and L orders: never
\* PACK_MISMATCH.
BuiltFor(d, p) ==
    /\ \A op \in {"A", "B"} : IsPk(p[op]) =>
          LET c == PlanCls(d, op) IN
          /\ p[op].s = d[op].s
          /\ SeqSet(p[op].K) = c.K /\ SeqSet(p[op].L) = c.L /\ SeqSet(p[op].R) = c.R
          /\ \A l \in AllLab(d) : (Ext(d, l) # 1 /\ l \notin Lab(d, "C")) => l \in Lab(d, op)
    /\ (IsPk(p.A) /\ IsPk(p.B)) => p.A.K = p.B.K /\ p.A.L = p.B.L
    \* no C-only label of extent 0 (C empty: a no-op). This lowering puts it
    \* in I with stride 0 in A (ClassOf), so post-swap A's slot key has an
    \* extra row mode (0, 0) and r = 0: a pack built by imd_pack_desc_for
    \* (X's labels only) is PACK_MISMATCH for that empty contraction.
    /\ \A l \in COnly(d) : Ext(d, l) = 1
PackComplete ==
    (pk # NoPk /\ BaseStatus(desc, xa, {Drop}) = "SUCCESS" /\ BuiltFor(desc, pk))
        => st # "PACK_MISMATCH"

\* imd_pack_desc_for (GIVEN, v1.5's orders of the unpacked plan) is
\* accepted and reproduces the v1.5 lowering exactly (orders and fusion).
PostRole(d, op) == IF DoSwap(d) THEN (IF op = "A" THEN "B" ELSE "A") ELSE op
DescFor(d, op) ==
    LET o == OrdInfoTop(d, NoPk).ord IN
    [op |-> op, s |-> d[op].s, K |-> o["K"], L |-> o["L"],
     R |-> o[IF PostRole(d, op) = "A" THEN "I" ELSE "J"]]
IsDescFor(d, p) == \A op \in {"A", "B"} : IsPk(p[op]) => p[op] = DescFor(d, op)
DescForReproduces ==
    (LowP /\ IsDescFor(desc, pk)) =>
        /\ st # "PACK_MISMATCH"
        /\ low.ord = OrdInfoTop(desc, NoPk).ord
        /\ low.fused = OrdInfoTop(desc, NoPk).fused

\* The adopted classes and orders (§6b.2): K, L and X's row class follow the
\* pack (K, L from post-swap A's when both are packed); the partner's row
\* class keeps v1.5's (CanonicalOrderExact). The pack's labels of the class
\* come in the pack's order; when the class is exactly the pack's (always,
\* unless an accepted plan never reads the pack: b·m·n·k = 0, a partner-
\* only label with k = 0 when the policy is dropped), the order IS the pack's.
AdoptedOrderExact ==
    (ok /\ pk # NoPk) =>
        LET kl == IF IsPk(low.pR["A"]) THEN low.pR["A"] ELSE low.pR["B"]
            want(g) == CASE g = "K" -> kl.K [] g = "L" -> kl.L
                         [] g = "I" -> low.pR["A"].R [] g = "J" -> low.pR["B"].R
        IN  /\ low.adopted = {"K", "L"} \cup {IF R = "A" THEN "I" ELSE "J" : R \in low.proles}
            /\ \A g \in low.adopted :
                  /\ SelectSeq(low.ord[g], LAMBDA l : l \in SeqSet(want(g)))
                       = SelectSeq(want(g), LAMBDA l : l \in low.grp[g])
                  /\ SeqSet(want(g)) = low.grp[g] => low.ord[g] = want(g)

\* AUTO (§6b.5: each class by (|s_X| after the diagonal merge, first
\* position in X's labels)) gives v1.5's K order whenever X is the post-swap
\* "big" operand with distinct K strides (§6b.2) and carries every K label.
FirstIn(sq, l) == CHOOSE p \in 1..Len(sq) : sq[p] = l /\ \A q \in 1..(p - 1) : sq[q] # l
AutoOrderV15 ==
    LowP =>
      \A op \in {o \in {"A", "B"} : IsPk(pk[o])} :
        LET Kp   == low.grp["K"]
            big  == IF low.m >= low.n THEN "A" ELSE "B"
            sx(l) == Abs(MS(desc, op, l))
            auto == SortBy(Kp, [l \in Kp |-> <<sx(l), FirstIn(desc[op].l, l)>>])
        IN  (PostRole(desc, op) = big /\ Kp \subseteq Lab(desc, op)
             /\ \A l1, l2 \in Kp : l1 # l2 => sx(l1) # sx(l2))
              => auto = OrdInfoTop(desc, NoPk).ord["K"]

\* Every pack byte K4 reads lies in the packed buffer (role B: rows r..n̂−1
\* of the last block are the tail).
PackNoOOB ==
    (ok /\ pk # NoPk /\ low.kind = "GEMM") =>
        \A R \in low.proles : low.pread[R] \subseteq DOMAIN low.pg[R]

\* §6b.4: the K padding and the tail of every pack are zero, its size is
\* b·r·k̂ + (⌈r⌉ − r)·k̂.
PackPaddingInert ==
    LowP =>
      \A R \in low.proles :
        LET pi == PkInfo(desc, low.pR[R]) IN
        /\ pi.size = pi.b * pi.r * pi.kh + (CeilQ(pi.r) - pi.r) * pi.kh
        /\ \A q \in DOMAIN pi.buf :
              pi.buf[q][1] <=> (q >= pi.b * pi.r * pi.kh \/ q % pi.kh >= pi.k)

\* The two policies of §6b.5: a partner-only K label (_CLASSES, §6b.10 #10)
\* and two packs with different K or L orders (_ORDER) are rejected ...
PackPolicy ==
    (LowP /\ ((\E R \in low.proles : low.partnerOnly[R]) \/ ~low.orderAgree)) => ~ok
\* ... and the key would reject a partner-only label too (it is a mode
\* (e, 0) of X in the slot key and k grows by e), except when k = 0 (an
\* extent-0 K label: the plan only scales C and never reads the pack).
KeyRejectsPartnerOnly ==
    LowP => \A R \in low.proles : (low.partnerOnly[R] /\ low.k # 0) => ~low.keyEq[R]

-----------------------------------------------------------------------------
(* Model well-formedness and coverage (checked by TLC as ASSUMEs).        *)

ASSUME Drop \in {"none", "Extents", "RepeatC", "COnly", "CStride", "Aliasing",
                 "Single", "Fusion", "NoSwap", "SwapFilter", "SwapLate",
                 "GemmLimit", "GemmLimitN", "GemmLimitDirect", "GemmLimitEmpty",
                 "Null", "WsAlias", "WsRangeBytes", "Misaligned", "Insufficient",
                 "OrderNullEmpty", "OrderEmptyLast", "OrderAlignFirst",
                 "OrderInsuffFirst", "OrderPreSwap", "PosSwapped", "GemmLimitK",
                 "NullStrict", "NullWsLax", "Kappa", "KappaDirect", "KappaEmpty",
                 "KappaK", "KappaTrunc",
                 \* D1
                 "PackFuseDefault", "KeyNoStrides", "KeyFuseCarriers", "PackNoAdopt",
                 "PackNoAdoptLower", "PackAfterKappa", "PackBStrideNh", "PackNoTail",
                 "PackPartnerOnly", "PackOrderCheck"}
ASSUME Packing \in {"none", "A", "B", "AB"} /\ PackEnum \in {"plan", "classes", "strides", "all"}
\* imd_contract_packed's execute-time checks are not modelled here: a packed
\* model uses default execute arguments (workspace far away and large enough).
ASSUME Packing # "none" =>
          \A x \in Execs : x.nul = {} /\ x.wsp[1] = "far" /\ x.dws >= 0
                           /\ x.mis = [X \in Tens |-> 0]
ASSUME GemmMax >= 1 /\ AlignWS >= 8 /\ AlignWS % 8 = 0
ASSUME \A x \in Execs :
          /\ x.mode \in {"EMULATE", "DIRECT"}
          /\ x.nul \subseteq {"A", "B", "C", "ws"} /\ x.a0 \in BOOLEAN
          /\ x.rn \in BOOLEAN /\ x.tn \in Nat /\ x.tn >= 1
          /\ x.mis \in [Tens -> 0..7]
          /\ x.wsp[1] \in Tens \cup {"far"} /\ x.wsp[2] \in {"after", "before"}
          /\ x.dws \in Int
ASSUME PadQ >= 1 /\ ExtentSet \subseteq 0..3 /\ ExtentSet # {}
\* CeilQ is the PadQ-roundup (k-hat, n-hat of §4a), incl. values that are
\* not multiples of PadQ and above one quantum.
ASSUME \A x \in 0..(3 * PadQ + 1) :
          CeilQ(x) % PadQ = 0 /\ x <= CeilQ(x) /\ CeilQ(x) < x + PadQ
ASSUME Labels = 1..Cardinality(Labels)
ASSUME \A r \in 0..3 : \A s \in (StridesA(r) \cup StridesB(r) \cup StridesC(r)) :
          DOMAIN s = 1..r

\* A descriptor lies in the enumerated space of this model.
InSpace(d) ==
    /\ Rk(d, "A") <= RankA /\ Rk(d, "B") <= RankB /\ Rk(d, "C") <= RankC
    /\ d.A.base \in BasesA /\ d.B.base \in BasesB /\ d.C.base \in BasesC
    /\ d.A.s \in StridesA(Rk(d, "A")) /\ d.B.s \in StridesB(Rk(d, "B"))
    /\ d.C.s \in StridesC(Rk(d, "C"))
    /\ AllLab(d) \subseteq Labels
    /\ ~CanonicalLabels \/ Canon(d.A.l \o d.B.l \o d.C.l)
    /\ \A X \in Tens : \A p \in Pos(d, X) : d[X].e[p] \in ExtentSet
    /\ PerModeExtents \/ ExtentsConsistent(d)
=============================================================================
