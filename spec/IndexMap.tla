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
(* IMPLEMENTATION MODEL (DESIGN §5 steps 1-10, §4 K2/K3/K4/K5):            *)
(*  status   step 1 extents (INVALID_VALUE), step 3 repeated C label       *)
(*           (NOT_SUPPORTED), step 7 C-only label with extent > 1          *)
(*           (NOT_SUPPORTED), the sorted-|stride| rule on C                *)
(*           (INVALID_VALUE), the C-vs-A/B interval test (ALIASING), in    *)
(*           this precedence order.                                        *)
(*  lower    step 2 diagonal merge (strides of a repeated label summed:    *)
(*           MS below), step 4 extent-1 labels dropped, step 5/6           *)
(*           classification by membership (A-only/B-only -> K with stride  *)
(*           0 in the other operand), step 9 canonical order per group,    *)
(*           step 8 fusion, step 10 additive scatter vectors               *)
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
(*  plan     NOOP if b*m*n = 0 (nothing read or written), SCALE if k = 0   *)
(*           (C <- beta*C: every C element written, A and B not read),     *)
(*           GEMM otherwise.                                               *)
(*                                                                         *)
(* CLAIMS (invariants; each comment names the DESIGN statement it checks): *)
(*  Partition, EinsumPreserved, ScatterExactlyOnce, ScatterCoversC,        *)
(*  AccessInBounds, ReadsExact, ZeroExtentRules, PaddingInert,             *)
(*  LayoutBijective, DecodeBijective, FusionPreservesAddresses,            *)
(*  OperandIndependence, NoCAliasing, StrideRuleImpliesInjective,          *)
(*  StatusCorrect (incl. the full precedence chain), CanonicalOrderSorted, *)
(*  OrderIndependent (expensive; small models only), and                   *)
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
(*  "Fusion"     §5 step 8      fusion condition checked in EVERY tensor   *)
(*                              carrying the group (dropped: only in the   *)
(*                              tensors where both labels appear, which is *)
(*                              the literal DESIGN wording)                *)
(*  "COnly"      §5 step 7      C-only label with extent > 1 rejected      *)
(*                              (dropping it is a POSITIVE model: the rule *)
(*                              is a v1 policy, broadcasting is correct)   *)
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
    MustCover        \* names of coverage descriptors (defined in MC) that
                     \*   this model must enumerate AND accept (ASSUME in MC)

VARIABLES
    desc,   \* the descriptor triple (chosen in Init, then constant)
    st,     \* its status
    low,    \* the lowering (a record), or "none" if rejected
    ein,    \* the einsum reference (a record), or "none" if rejected
    el      \* el[X] = Elems(desc, X), cached

vars == <<desc, st, low, ein, el>>

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
(* Status (DESIGN §5 lowering order; ALIASING is checked last, at execute *)
(* time). The C rule applies only to a non-empty C (see design_issues).   *)

Status(d) ==
    IF Drop # "Extents" /\ ~ExtentsConsistent(d)            THEN "INVALID_VALUE"
    ELSE IF Drop # "RepeatC" /\ RepeatedC(d)                 THEN "NOT_SUPPORTED"
    ELSE IF Drop # "COnly" /\ COnlyBig(d)                    THEN "NOT_SUPPORTED"
    ELSE IF Drop # "CStride" /\ ~Empty(d, "C") /\ ~StrideRule(d, "C")
                                                             THEN "INVALID_VALUE"
    ELSE IF Drop # "Aliasing" /\ IntervalAlias(d)            THEN "ALIASING"
    ELSE "SUCCESS"

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

Lower(d) ==
    LET labs == AllLab(d)
        ext  == [l \in labs |-> Ext(d, l)]
        ms   == [X \in Tens, l \in labs |-> MS(d, X, l)]
        kept == {l \in labs : ext[l] # 1}                         \* step 4
        cls  == [l \in kept |-> ClassOf(d, l)]
        grp  == [g \in Grps |-> {l \in kept : cls[l] = g}]
        sz   == [g \in Grps |-> ProdOver(grp[g], ext)]
        m == sz["I"]  n == sz["J"]  k == sz["K"]  b == sz["L"]
        kh == CeilQ(k)  nh == CeilQ(n)
        big == IF m >= n THEN "A" ELSE "B"                        \* step 9
        oth == IF big = "A" THEN "B" ELSE "A"
        key == [g \in Grps |-> [l \in grp[g] |->
                  LET a == Abs(ms["A", l])  bb == Abs(ms["B", l])
                      c == Abs(ms["C", l])  f == FirstPos(d, l)
                  IN  CASE g = "I" -> <<c, a, 0, f>>
                        [] g = "J" -> <<c, bb, 0, f>>
                        [] g = "L" -> <<c, a, bb, f>>
                        [] g = "K" -> <<Abs(ms[big, l]), Abs(ms[oth, l]), 0, f>>]]
        ord  == [g \in Grps |-> SortBy(grp[g], key[g])]
        \* step 8: g_t fuses into g_(t-1) iff s_X(g_t) = s_X(g_(t-1))*e(g_(t-1))
        \* for every X in Chk.
        fuseOK(g, a, c) ==
            LET chk == IF Drop = "Fusion"
                       THEN {X \in Car[g] : a \in Lab(d, X) /\ c \in Lab(d, X)}
                       ELSE Car[g]
            IN  \A X \in chk : ms[X, c] = ms[X, a] * ext[a]
        unf  == [g \in Grps |-> [t \in 1..Len(ord[g]) |->
                   [e |-> ext[ord[g][t]], s |-> [X \in Tens |-> ms[X, ord[g][t]]]]]]
        fused == [g \in Grps |->
                   LET o == ord[g]
                       S == {t \in 1..Len(o) : t = 1 \/ ~fuseOK(g, o[t - 1], o[t])}
                       st0 == SortBy(S, [t \in S |-> <<t>>])
                       ns == Len(st0)
                       last(j) == IF j < ns THEN st0[j + 1] - 1 ELSE Len(o)
                   IN  [j \in 1..ns |->
                         [e |-> ProdOver(st0[j]..last(j), [t \in 1..Len(o) |-> ext[o[t]]]),
                          s |-> [X \in Tens |-> ms[X, o[st0[j]]]]]]]
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
        kind == IF b * m * n = 0 THEN "NOOP" ELSE IF k = 0 THEN "SCALE" ELSE "GEMM"
        \* K4 over the padded box; K5 scatters columns j < n only.
        gdom == {<<lam, i, j, h>> : lam \in 0..(b - 1), i \in 0..(m - 1),
                                     j \in 0..(nh - 1), h \in 0..(kh - 1)}
        ca(t) == bufA[(t[1] * m + t[2]) * kh + t[4]]
        cb(t) == bufB[(t[1] * nh + t[3]) * kh + t[4]]
        nz   == {t \in gdom : ~ca(t)[1] /\ ~cb(t)[1]}
        cadr(t) == IF t[3] < n THEN offC(t[1], t[2], t[3]) ELSE Unscattered
        contrib == IF kind = "GEMM"
                   THEN MkBag(nz, [t \in nz |-> <<cadr(t), ca(t)[2], cb(t)[2]>>])
                   ELSE EmptyBag
        wdom == {<<lam, i, j>> : lam \in 0..(b - 1), i \in 0..(m - 1), j \in 0..(n - 1)}
        writes == IF kind = "NOOP" THEN EmptyBag
                  ELSE MkBag(wdom, [w \in wdom |-> offC(w[1], w[2], w[3])])
        readsA == IF kind = "GEMM" THEN {bufA[q][2] : q \in {q \in DOMAIN bufA : ~bufA[q][1]}} ELSE {}
        readsB == IF kind = "GEMM" THEN {bufB[q][2] : q \in {q \in DOMAIN bufB : ~bufB[q][1]}} ELSE {}
    IN  [labs |-> labs, ext |-> ext, kept |-> kept, cls |-> cls, grp |-> grp,
         sz |-> sz, m |-> m, n |-> n, k |-> k, b |-> b, kh |-> kh, nh |-> nh,
         ord |-> ord, unf |-> unf, fused |-> fused, vec |-> vec, vecU |-> vecU,
         dec |-> dec, bufA |-> bufA, bufB |-> bufB, kind |-> kind,
         contrib |-> contrib, writes |-> writes, readsA |-> readsA, readsB |-> readsB]

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
         readsA |-> {addr("A", x) : x \in pts},
         readsB |-> {addr("B", x) : x \in pts},
         \* C output positions (assignments of C's labels): one write each
         cpos   |-> MkBag(cpts, [y \in cpts |-> addr("C", y)]),
         pts    |-> pts]

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
    /\ st  = Status(desc)
    /\ low = IF st = "SUCCESS" THEN Lower(desc) ELSE "none"
    /\ ein = IF st = "SUCCESS" /\ ExtentsConsistent(desc) THEN Einsum(desc) ELSE "none"
    /\ el  = [X \in Tens |-> Elems(desc, X)]

Next == UNCHANGED vars
Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Invariants. "ok" = the descriptor was accepted.                        *)

ok == st = "SUCCESS"

\* DESIGN §5 "classification is a partition": the kept labels (extent # 1)
\* are split into I, J, K, L, each with the right membership set.
Partition ==
    ok => /\ UNION {low.grp[g] : g \in Grps} = low.kept
          /\ \A g, h \in Grps : g # h => low.grp[g] \cap low.grp[h] = {}
          /\ \A l \in low.kept :
               LET M == {X \in Tens : l \in Lab(desc, X)} IN
               CASE low.cls[l] = "I" -> M = {"A", "C"}
                                          \/ (M = {"C"} /\ (low.ext[l] = 0 \/ Drop = "COnly"))
                 [] low.cls[l] = "J" -> M = {"B", "C"}
                 [] low.cls[l] = "K" -> M \in {{"A", "B"}, {"A"}, {"B"}}
                 [] low.cls[l] = "L" -> M = Tens
                 [] OTHER            -> FALSE

\* DESIGN §5 "lowering preserves the einsum" and "Scatter o GEMM o Gather =
\* einsum": the packed, padded, fused, reordered GEMM view produces exactly
\* the einsum's multiset of (addrC, addrA, addrB) contributions. A nonzero
\* product in an unscattered column would appear with addrC = Unscattered.
EinsumPreserved == ok => low.contrib = ein.bag

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
    ok => /\ low.readsA \subseteq el["A"]
          /\ low.readsB \subseteq el["B"]
          /\ DOMAIN low.writes \subseteq el["C"]

\* The gather reads exactly the elements the einsum uses (no more, no less).
ReadsExact ==
    ok /\ low.kind = "GEMM" => low.readsA = ein.readsA /\ low.readsB = ein.readsB

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
                        SumOver(G, [l \in G |-> low.dec[g][q][l] * MS(desc, X, l)])

\* DESIGN §5 step 8: fusion preserves every address: the scatter vectors
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

\* The rejection reasons in DESIGN's precedence order (§5 lowering order:
\* step 1 extents, step 3 repeated C label, step 7 C-only label, then the
\* §5-table C self-overlap rule, then the execute-time ALIASING test):
\* <<rule name (the Drop that disables it), fires?, status>>. Written as a
\* table, independently of the IF-chain in Status, so a reordering of
\* Status is caught by StatusCorrect.
Reasons(d) ==
    << <<"Extents",  ~ExtentsConsistent(d),                 "INVALID_VALUE">>,
       <<"RepeatC",  RepeatedC(d),                          "NOT_SUPPORTED">>,
       <<"COnly",    COnlyBig(d),                           "NOT_SUPPORTED">>,
       <<"CStride",  ~Empty(d, "C") /\ ~StrideRule(d, "C"), "INVALID_VALUE">>,
       <<"Aliasing", IntervalAlias(d),                      "ALIASING">> >>
\* The status of the FIRST enabled reason that fires; SUCCESS if none.
ExpectedStatus(d) ==
    LET R == Reasons(d)
        F == {t \in 1..Len(R) : R[t][1] # Drop /\ R[t][2]}
    IN  IF F = {} THEN "SUCCESS" ELSE R[CHOOSE t \in F : \A u \in F : t <= u][3]

\* DESIGN §5 / §6 statuses: each rejection has the right status and reason
\* and comes in DESIGN's precedence order (first firing reason wins, e.g.
\* C[i,i] with strides (1,1) is NOT_SUPPORTED, not INVALID_VALUE);
\* acceptance implies every SEMANTIC legality fact (not just the cheap
\* tests that imply them).
StatusCorrect ==
    /\ st \in {"SUCCESS", "INVALID_VALUE", "NOT_SUPPORTED", "ALIASING"}
    /\ st = ExpectedStatus(desc)
    /\ ~ExtentsConsistent(desc) => st = "INVALID_VALUE"
    /\ st = "NOT_SUPPORTED" => ExtentsConsistent(desc) /\ Unsupported(desc)
    /\ st = "INVALID_VALUE" => ~ExtentsConsistent(desc)
                               \/ (~Empty(desc, "C") /\ ~StrideRule(desc, "C"))
    /\ st = "ALIASING" => IntervalAlias(desc)
    /\ ok => /\ ExtentsConsistent(desc)
             /\ ~Unsupported(desc)
             /\ Injective(desc, "C")
             /\ el["C"] \cap (el["A"] \cup el["B"]) = {}
    \* A aliasing B is never a reason to reject.
    /\ (ExtentsConsistent(desc) /\ ~Unsupported(desc)
        /\ (Empty(desc, "C") \/ StrideRule(desc, "C")) /\ ~IntervalAlias(desc)) => ok

\* DESIGN §5 step 9 (and §1.1: the k-bar order is a fixed function of the
\* descriptors): within I, J and L the canonical order is non-decreasing in
\* C's merged |stride| (so CRT scatter writes coalesce); within K it is
\* non-decreasing in the LARGER operand's |stride| (A if m >= n, else B;
\* stride 0 for a label the operand lacks). Ties are broken by a fixed key
\* (see Lower) that DESIGN does not prescribe, so only the DESIGN key is
\* checked. The "larger operand" is recomputed here from m, n.
CanonicalOrderSorted ==
    ok => LET bigX == IF low.m >= low.n THEN "A" ELSE "B"
              keyT(g) == IF g = "K" THEN bigX ELSE "C"
          IN  \A g \in Grps : \A t \in 1..(Len(low.ord[g]) - 1) :
                Abs(MS(desc, keyT(g), low.ord[g][t]))
                  <= Abs(MS(desc, keyT(g), low.ord[g][t + 1]))

\* DESIGN §1.1 / r_layout §4.1: the result does not depend on the group
\* orders (canonical orders are a performance choice only). Every
\* permutation of every group, unfused, gives the einsum multiset.
Perms(S) == {f \in [1..Cardinality(S) -> S] : \A i, j \in DOMAIN f : i # j => f[i] # f[j]}
OrderIndependent ==
    (ok /\ CheckOrders) =>
      \A pI \in Perms(low.grp["I"]), pJ \in Perms(low.grp["J"]),
         pK \in Perms(low.grp["K"]), pL \in Perms(low.grp["L"]) :
        LET o == [I |-> pI, J |-> pJ, K |-> pK, L |-> pL]
            md(g) == [t \in 1..Len(o[g]) |->
                        [e |-> low.ext[o[g][t]],
                         s |-> [X \in Tens |-> MS(desc, X, o[g][t])]]]
            v == [X \in Tens, g \in Grps |-> RadixVec(md(g), X, low.sz[g])]
            dom == {<<lam, i, j, h>> : lam \in 0..(low.b - 1), i \in 0..(low.m - 1),
                                        j \in 0..(low.n - 1), h \in 0..(low.k - 1)}
            tr(t) == <<desc.C.base + v["C", "L"][t[1]] + v["C", "I"][t[2]] + v["C", "J"][t[3]],
                       desc.A.base + v["A", "L"][t[1]] + v["A", "I"][t[2]] + v["A", "K"][t[4]],
                       desc.B.base + v["B", "L"][t[1]] + v["B", "K"][t[4]] + v["B", "J"][t[3]]>>
        IN  MkBag(dom, [t \in dom |-> tr(t)]) = ein.bag

-----------------------------------------------------------------------------
(* Model well-formedness and coverage (checked by TLC as ASSUMEs).        *)

ASSUME Drop \in {"none", "Extents", "RepeatC", "COnly", "CStride", "Aliasing",
                 "Single", "Fusion"}
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
