---------------------------- MODULE MC_IndexMap ----------------------------
(***************************************************************************)
(* Model values for IndexMap (.cfg files cannot hold negative literals,   *)
(* tuples or operators, so they live here and the cfgs use               *)
(* `Const <- Def`).                                                        *)
(*                                                                         *)
(* STRIDE PATTERNS (r = rank; mode p = 1..r):                              *)
(*  Col(r)    {<<1, 2, 4>>[1..r]}: dense, first mode fastest. With extent  *)
(*            2 every distinct-label index tuple has a distinct address,   *)
(*            so the contribution multiset encodes the full index map (a  *)
(*            lost, duplicated or mis-strided term changes the bag), and   *)
(*            consecutive modes are fusable (2 = 1*2, 4 = 2*2).            *)
(*  ColRow(r) Col plus the reversed (row-major) pattern.                   *)
(*  Free5(r)  every stride in {-2,-1,0,1,2}: zero and negative strides,    *)
(*            fusion ON its boundary (s2 = s1*e1, e.g. (1,2), (-1,-2),     *)
(*            (0,0)) and one off it (e.g. (1,1), (2,1), (1,-2)).           *)
(*  CStr(r)   every stride in -3..4: the sorted-|stride| rule both ON its *)
(*            boundary (|s_t| = span + 1, e.g. (1,2,4) at extent 2) and   *)
(*            one below it (|s_t| = span, e.g. (1,2,3): 3 = 1+2 collides). *)
(*  Al4(r)    every stride in {-2,-1,1,2}, for the aliasing model.        *)
(*  Dn3(r)    ColRow plus the dense-at-extent-3 patterns (1,3) and (3,1):  *)
(*            fusion ON its boundary at extent 3 (s2 = 3*s1), so fused     *)
(*            modes of extent 6 = 3*2 / 2*3 (mixed radices) occur.          *)
(*                                                                         *)
(* SCALING. Real: rank <= 64, int64 extents/strides, PadQ = 16. Here:     *)
(* rank <= 3, extents 0..3, |stride| <= 4, PadQ = 2 or 4. Extent 0 is     *)
(* empty and 1 is dropped; a real mode is checked at BOTH 2 and 3, because *)
(* at 2 every (e-1) factor is 1 and k, n are powers of two: only extent 3  *)
(* exercises the (e-1) span of the stride rule and of the interval test,   *)
(* k, n that are not multiples of PadQ (k = 3, 6, 9 -> kh = 4, 6, 10:      *)
(* padded; k = 2, 4 -> exact, ON the boundary) and mixed radices.          *)
(*                                                                         *)
(* POSITIVE models (must pass; MustCover names descriptors each one is   *)
(* REQUIRED, by ASSUME, to enumerate and accept, see Cov below):           *)
(*  pos_struct   4 labels, ranks <= 3/3/3, extent 2, Col strides: every   *)
(*               label structure up to renaming (all 16 I/J/K/L emptiness *)
(*               patterns, diagonals A[k,k], A-only and B-only labels,    *)
(*               C-only and repeated-C rejections), A and B at the same   *)
(*               base (A aliases B: allowed). 26,454 descriptors.         *)
(*  pos_extents  3 labels, ranks <= 3/3/2, extents {0,1,2} per label, Col  *)
(*               strides, PadQ = 4: NOOP (empty I/J/L), SCALE (k = 0),     *)
(*               extent-1 dropping (incl. C-only extent 1), padding.       *)
(*  pos_mismatch 2 labels, ranks <= 2, extents {0,1,2} PER MODE: every     *)
(*               mismatch (1 vs 2, 0 vs 1, within one tensor or across)    *)
(*               -> INVALID_VALUE; equal extents (ON the boundary) pass.   *)
(*  pos_cstride  3 labels, ranks <= 2/1/3, C strides -3..4, extent 2: the  *)
(*               sorted-|stride| rule ON (|s_t| = span+1) and one below    *)
(*               (|s_t| = span) its boundary, ties, signs, zero strides.   *)
(*  pos_cstride_e1  C alone (A, B scalars), rank <= 3, strides -3..4,      *)
(*               extents {1,2}, Drop = "COnly" so that every C label is a *)
(*               (legal, broadcast) C-only label and acceptance depends on *)
(*               the stride rule alone, incl. extent-1 modes of any stride.*)
(*  pos_alias    2 labels, ranks <= 1/1/2, A strides {-2..2}, C strides    *)
(*               {-2,-1,1,2}, extents {0,2}, A base {0,2} (a lowering that *)
(*               drops A's base offset breaks EinsumPreserved), B base     *)
(*               {0,3}, C base -3..4:                                      *)
(*               the interval test incl. touching half-open intervals,     *)
(*               empty tensors and negative strides.                       *)
(*  pos_fusion   3 labels, ranks <= 2/2/1, A and B strides {-2..2}, extent *)
(*               2: fusion ON (s2 = s1*e1, incl. negative and zero) and    *)
(*               OFF its boundary, canonical K order with ties, stride-0   *)
(*               lowering of single-operand labels inside fused groups.    *)
(*  pos_orders   3 labels, ranks <= 3/3/2, ColRow, extent 2, CheckOrders:  *)
(*               the result is independent of every group order.           *)
(*  pos_ext3     3 labels, ranks <= 2/1/2, extents {2,3} per label, A      *)
(*               strides Dn3, C strides -3..4, PadQ = 2: the stride rule's *)
(*               (e-1) span ON (C e=(3,2), s=(1,3)) and one below          *)
(*               (s=(1,2)) its boundary, mixed radices, fusion at extent 3 *)
(*               (fused extent 6), k = 3, 6, 9 padded to 4, 6, 10.         *)
(*  pos_alias3   2 labels, ranks <= 1/1/1, extents {0,3}, A strides         *)
(*               {-2..2}, A base {0,2}, B base {0,3}, C base -3..4: the     *)
(*               interval test with (e-1)*s spans (A = [-4,1) for s = -2),  *)
(*               touching both ends, k = 3 -> kh = 4.                      *)
(*  pos_conly    Drop = "COnly", 3 labels, ranks <= 2, extents {0,1,2}:    *)
(*               the C-only rule is NOT needed for correctness (broadcast  *)
(*               lowering to I with stride 0 in A is exact); it is a v1    *)
(*               policy.                                                   *)
(*                                                                         *)
(* NEGATIVE models (must fail as their "\* EXPECT:" says):                 *)
(*  neg_cstride  Drop "CStride": a C with |s_t| = span (one below the      *)
(*               boundary) is accepted and written twice.                  *)
(*  neg_extents  Drop "Extents": a mismatched label is lowered with one    *)
(*               extent, so some tensor is read/written out of bounds.     *)
(*  neg_single   Drop "Single": an A-only/B-only label is not summed.      *)
(*  neg_fusion   Drop "Fusion" (DESIGN step 8 read literally: check only   *)
(*               tensors where both labels appear): a stride-0 lowered     *)
(*               operand is fused wrongly.                                 *)
(*  neg_repeatc  Drop "RepeatC": C[i,i] writes only its diagonal, so the   *)
(*               off-diagonal elements of C never get beta*C.             *)
(*  neg_alias    Drop "Aliasing": C shares an element with A.             *)
(*  neg_rule_sufficient_only  StrideRuleNecessary fails: the rule rejects  *)
(*               an injective C (e = (2,2,2), s = (2,3,4)), as DESIGN says.*)
(***************************************************************************)
EXTENDS IndexMap

Col(r)    == {[p \in 1..r |-> 2^(p - 1)]}
ColRow(r) == {[p \in 1..r |-> 2^(p - 1)], [p \in 1..r |-> 2^(r - p)]}
Free5(r)  == [1..r -> -2..2]
CStr(r)   == [1..r -> -3..4]
Al4(r)    == [1..r -> {-2, -1, 1, 2}]
Dn3(r)    == ColRow(r) \cup {[p \in 1..r |-> 3^(p - 1)], [p \in 1..r |-> 3^(r - p)]}

Zero      == {0}
B0        == {0}
B03       == {0, 3}
B02       == {0, 2}
C100      == {100}
C50       == {50}
CAl       == -3..4

-----------------------------------------------------------------------------
(* Boundary sentinels (constant-level; TLC evaluates them as ASSUMEs).     *)

T(bs, l, e, s) == [base |-> bs, l |-> l, e |-> e, s |-> s]
Scalar(bs)     == T(bs, <<>>, <<>>, <<>>)
D(a, b, c)     == [A |-> a, B |-> b, C |-> c]

\* C over three extent-2 modes; only the C part matters for StrideRule.
C3(s) == D(Scalar(0), Scalar(0), T(100, <<1, 2, 3>>, <<2, 2, 2>>, s))

\* ON the boundary: |s_t| = span + 1 at every t -> accepted and injective.
ASSUME StrideRule(C3(<<1, 2, 4>>), "C") /\ Injective(C3(<<1, 2, 4>>), "C")
ASSUME StrideRule(C3(<<-4, 1, -2>>), "C") /\ Injective(C3(<<-4, 1, -2>>), "C")
\* One below (|s_3| = span = 3): rejected and really NOT injective.
ASSUME ~StrideRule(C3(<<1, 2, 3>>), "C") /\ ~Injective(C3(<<1, 2, 3>>), "C")
\* Equal |strides| (tie, opposite signs): rejected, not injective.
ASSUME ~StrideRule(C3(<<1, -1, 4>>), "C") /\ ~Injective(C3(<<1, -1, 4>>), "C")
\* Only sufficient: injective (offsets 0,2,3,5,4,6,7,9) but rejected.
ASSUME ~StrideRule(C3(<<2, 3, 4>>), "C") /\ Injective(C3(<<2, 3, 4>>), "C")
\* Extent-1 modes are ignored by the rule (stride 0 allowed there).
ASSUME StrideRule(D(Scalar(0), Scalar(0), T(0, <<1, 2>>, <<1, 2>>, <<0, 1>>)), "C")

\* At extent 3 the (e-1) factor matters (at extent 2 it is 1).
C2(e, s) == D(Scalar(0), Scalar(0), T(100, <<1, 2>>, e, s))
\* ON the boundary: |s_2| = 3 = |s_1|*(3-1) + 1 -> accepted, injective.
ASSUME StrideRule(C2(<<3, 2>>, <<1, 3>>), "C") /\ Injective(C2(<<3, 2>>, <<1, 3>>), "C")
ASSUME StrideRule(C2(<<3, 3>>, <<1, 3>>), "C") /\ Injective(C2(<<3, 3>>, <<1, 3>>), "C")
\* Sorting is by |stride|, not by mode order: modes (e=2,s=3), (e=3,s=1).
ASSUME StrideRule(C2(<<2, 3>>, <<3, 1>>), "C") /\ Injective(C2(<<2, 3>>, <<3, 1>>), "C")
\* One below (|s_2| = 2 = span): rejected and really NOT injective
\* (C[2,0] and C[0,1] share offset 2). A rule without (e-1) accepts it.
ASSUME ~StrideRule(C2(<<3, 2>>, <<1, 2>>), "C") /\ ~Injective(C2(<<3, 2>>, <<1, 2>>), "C")
\* Only sufficient at extent 3 (r_layout): offsets 0,2,4,3,5,7 distinct.
ASSUME ~StrideRule(C2(<<3, 2>>, <<2, 3>>), "C") /\ Injective(C2(<<3, 2>>, <<2, 3>>), "C")

\* Intervals: A = [0, 4) (e = (2,2), s = (1,2)); C scalar at 4 touches it
\* (half-open: no alias), at 3 it is inside (alias); negative strides
\* A = [-3, 1); an empty A never aliases.
AI(s, e) == T(0, <<1, 2>>, e, s)
ASSUME ~IntervalAlias(D(AI(<<1, 2>>, <<2, 2>>), Scalar(50), Scalar(4)))
ASSUME  IntervalAlias(D(AI(<<1, 2>>, <<2, 2>>), Scalar(50), Scalar(3)))
ASSUME  IntervalAlias(D(AI(<<-1, -2>>, <<2, 2>>), Scalar(50), Scalar(-3)))
ASSUME ~IntervalAlias(D(AI(<<-1, -2>>, <<2, 2>>), Scalar(50), Scalar(1)))
ASSUME ~IntervalAlias(D(AI(<<1, 2>>, <<2, 0>>), Scalar(50), Scalar(0)))
\* Extent 3, stride -2: A = {0,-2,-4}, interval [-4, 1) (the (e-1)*s span).
A3 == T(0, <<1>>, <<3>>, <<-2>>)
ASSUME  IntervalAlias(D(A3, Scalar(50), Scalar(-4)))
ASSUME ~IntervalAlias(D(A3, Scalar(50), Scalar(1)))
ASSUME ~IntervalAlias(D(A3, Scalar(50), Scalar(-5)))
\* Conservative: -3 is inside the interval but not an element of A.
ASSUME  IntervalAlias(D(A3, Scalar(50), Scalar(-3))) /\ -3 \notin Elems(D(A3, Scalar(50), Scalar(-3)), "A")

\* Status precedence (DESIGN lowering order): mismatch beats repeated C.
ASSUME Status(D(T(0, <<1>>, <<1>>, <<1>>), Scalar(50),
                T(100, <<1, 1>>, <<2, 2>>, <<1, 2>>))) =
       (IF Drop = "Extents" THEN "NOT_SUPPORTED" ELSE "INVALID_VALUE")
\* Repeated C beats the C rule: C[1,1] with strides (1,1) fails both.
ASSUME Status(D(T(0, <<1>>, <<2>>, <<1>>), Scalar(50),
                T(100, <<1, 1>>, <<2, 2>>, <<1, 1>>))) =
       (IF Drop = "RepeatC" THEN "INVALID_VALUE" ELSE "NOT_SUPPORTED")
\* C-only (label 2, extent 2) beats the C rule: strides (1,1) fail it.
ASSUME Status(D(T(0, <<1>>, <<2>>, <<1>>), Scalar(50),
                T(100, <<1, 2>>, <<2, 2>>, <<1, 1>>))) =
       (IF Drop = "COnly" THEN "INVALID_VALUE" ELSE "NOT_SUPPORTED")
\* The C rule beats ALIASING: C = [0,3) fails the rule and overlaps A.
ASSUME Status(D(T(0, <<1>>, <<2>>, <<1>>), T(0, <<2>>, <<2>>, <<1>>),
                T(0, <<1, 2>>, <<2, 2>>, <<1, 1>>))) =
       (IF Drop = "CStride" THEN "ALIASING" ELSE "INVALID_VALUE")

-----------------------------------------------------------------------------
(* Witnesses for the negative models: the exact descriptor each dropped  *)
(* rule mishandles, and (in every Drop = "none" model) proof that the     *)
(* design handles the same descriptor correctly. This pins each negative  *)
(* model's failure to the intended cause.                                 *)

\* Label 1 in A and B (K), label 2 only in A (A-only -> K, stride 0 in B).
\* A = [1,2] strides (1,2): dense, so 1 and 2 fuse in A; B = [1] stride 1.
\* Literal step 8 ("tensors where both appear" = A only) fuses them and
\* gives B the fused stride 1 for label 2 as well: B is read at 0..3.
WFuse == D(T(0, <<1, 2>>, <<2, 2>>, <<1, 2>>), T(0, <<1>>, <<2>>, <<1>>), Scalar(100))
ASSUME Status(WFuse) = "SUCCESS"
ASSUME Drop = "Fusion" => Lower(WFuse).contrib # Einsum(WFuse).bag
\* The same descriptor without step 6: label 2 is not summed.
ASSUME Drop = "Single" => Lower(WFuse).contrib # Einsum(WFuse).bag
ASSUME Drop = "none"   => Lower(WFuse).contrib = Einsum(WFuse).bag

\* C[1,2] with strides (1,1): |s_2| = span = 1, one below the boundary.
WCStr == D(T(0, <<1>>, <<2>>, <<1>>), T(0, <<2>>, <<2>>, <<1>>),
           T(100, <<1, 2>>, <<2, 2>>, <<1, 1>>))
ASSUME Drop # "CStride" => Status(WCStr) = "INVALID_VALUE"
ASSUME Drop = "CStride" => \E a \in DOMAIN Lower(WCStr).writes : Lower(WCStr).writes[a] = 2

\* C[1,1] (repeated C label), strides (1,2): only the diagonal is written.
WRepC == D(T(0, <<1>>, <<2>>, <<1>>), T(0, <<1>>, <<2>>, <<1>>),
           T(100, <<1, 1>>, <<2, 2>>, <<1, 2>>))
ASSUME Drop # "RepeatC" => Status(WRepC) = "NOT_SUPPORTED"
ASSUME Drop = "RepeatC" => DOMAIN Lower(WRepC).writes = {100, 103}
                           /\ Elems(WRepC, "C") = {100, 101, 102, 103}

\* Label 1 has extent 1 in A, 2 in B: lowered with 2, A is read at 1.
WExt == D(T(0, <<1>>, <<1>>, <<1>>), T(0, <<1>>, <<2>>, <<1>>), Scalar(100))
ASSUME Drop # "Extents" => Status(WExt) = "INVALID_VALUE"
ASSUME Drop = "Extents" => 1 \in Lower(WExt).readsA /\ Elems(WExt, "A") = {0}

\* C (scalar at 1) is element 1 of A = [1] (stride 1, extent 2).
WAl == D(T(0, <<1>>, <<2>>, <<1>>), T(50, <<1>>, <<2>>, <<1>>), Scalar(1))
ASSUME Drop # "Aliasing" => Status(WAl) = "ALIASING"
ASSUME Drop = "Aliasing" => Status(WAl) = "SUCCESS" /\ 1 \in Elems(WAl, "A")

\* C-only label 2 (broadcast) of extent 2: rejected in v1; with the rule
\* dropped, the lowering (I, stride 0 in A) is exact.
WCOnly == D(T(0, <<1>>, <<2>>, <<1>>), T(0, <<1>>, <<2>>, <<1>>),
            T(100, <<2>>, <<2>>, <<1>>))
ASSUME Drop # "COnly" => Status(WCOnly) = "NOT_SUPPORTED"
ASSUME Drop = "COnly" => /\ Lower(WCOnly).contrib = Einsum(WCOnly).bag
                         /\ DOMAIN Lower(WCOnly).writes = {100, 101}

-----------------------------------------------------------------------------
(* Coverage: each model must enumerate AND accept the descriptors named in *)
(* its MustCover, so no positive model passes vacuously.                   *)

Cov == [
  \* C[i,j,l] = sum_k A[i,k,l] B[k,j,l]: all four groups non-empty, dense.
  gemm4    |-> D(T(0, <<1, 2, 3>>, <<2, 2, 2>>, <<1, 2, 4>>),
                 T(0, <<2, 4, 3>>, <<2, 2, 2>>, <<1, 2, 4>>),
                 T(100, <<1, 4, 3>>, <<2, 2, 2>>, <<1, 2, 4>>)),
  \* diagonal A[k,k,a] with an A-only label a, B[k,j], C[j].
  diag     |-> D(T(0, <<1, 1, 2>>, <<2, 2, 2>>, <<1, 2, 4>>),
                 T(0, <<1, 3>>, <<2, 2>>, <<1, 2>>),
                 T(100, <<3>>, <<2>>, <<1>>)),
  \* k = 0 (K label of extent 0): SCALE, C <- beta*C.
  kzero    |-> D(T(0, <<1, 2>>, <<2, 0>>, <<1, 2>>),
                 T(0, <<2, 3>>, <<0, 2>>, <<1, 2>>),
                 T(100, <<1, 3>>, <<2, 2>>, <<1, 2>>)),
  \* m = 0 (I label of extent 0): NOOP; plus an extent-1 C-only label.
  mzero    |-> D(T(0, <<1, 2>>, <<0, 2>>, <<1, 2>>),
                 T(0, <<2>>, <<2>>, <<1>>),
                 T(100, <<1, 3>>, <<0, 1>>, <<1, 2>>)),
  \* fusion ON its boundary with negative strides in A and zero in B.
  fuseneg  |-> D(T(0, <<1, 2>>, <<2, 2>>, <<-1, -2>>),
                 T(0, <<1, 2>>, <<2, 2>>, <<0, 0>>),
                 Scalar(100)),
  \* fusion one OFF its boundary (A strides (1,1)).
  fuseoff  |-> D(T(0, <<1, 2>>, <<2, 2>>, <<1, 1>>),
                 T(0, <<1, 2>>, <<2, 2>>, <<1, 2>>),
                 Scalar(100)),
  \* C rule ON its boundary at rank 3 with mixed signs: s = (4, 1, -2).
  crule3   |-> D(T(0, <<1, 2>>, <<2, 2>>, <<1, 2>>),
                 T(0, <<3>>, <<2>>, <<1>>),
                 T(50, <<1, 2, 3>>, <<2, 2, 2>>, <<4, 1, -2>>)),
  \* broadcast C-only labels (needs Drop = "COnly"), rule ON boundary.
  bcast3   |-> D(Scalar(0), Scalar(0),
                 T(50, <<1, 2, 3>>, <<2, 2, 2>>, <<1, 2, 4>>)),
  \* extent-1 C mode with stride 0 (ignored by the rule).
  ce1zero  |-> D(Scalar(0), Scalar(0),
                 T(50, <<1, 2, 3>>, <<2, 1, 2>>, <<1, 0, 2>>)),
  \* half-open intervals touching: A = [0,3), B = [0,2) (B aliases A), C at 3.
  touch    |-> D(T(0, <<1>>, <<2>>, <<2>>), T(0, <<1>>, <<2>>, <<1>>), Scalar(3)),
  \* empty A (extent 0) overlapping C's address range: no alias.
  emptyA   |-> D(T(0, <<1>>, <<0>>, <<1>>), T(3, <<2>>, <<2>>, <<1>>),
                 T(0, <<2>>, <<2>>, <<1>>)),
  \* mismatch model: a consistent descriptor with extents 0, 1, 2.
  mixed    |-> D(T(0, <<1, 2>>, <<2, 1>>, <<1, 2>>),
                 T(0, <<1>>, <<2>>, <<1>>),
                 T(100, <<2>>, <<1>>, <<1>>)),
  \* row-major operands (ColRow): orders differ from mode order.
  rows     |-> D(T(0, <<1, 2>>, <<2, 2>>, <<2, 1>>),
                 T(0, <<2, 3>>, <<2, 2>>, <<2, 1>>),
                 T(100, <<1, 3>>, <<2, 2>>, <<2, 1>>)),
  \* extent 3: C e=(3,2), s=(1,3) ON the (e-1) boundary; A-only label 2
  \* of extent 3 gives k = 3, kh = 4 (PadQ 2); m = 3, n = 2.
  ext3rule |-> D(T(0, <<1, 2>>, <<3, 3>>, <<1, 3>>),
                 T(0, <<3>>, <<2>>, <<1>>),
                 T(50, <<1, 3>>, <<3, 2>>, <<1, 3>>)),
  \* fusion ON its boundary at extent 3: I labels 1 (e 3), 2 (e 2) with
  \* strides (1,3) in A and C fuse into one mode of extent 6.
  fuse3    |-> D(T(0, <<1, 2>>, <<3, 2>>, <<1, 3>>),
                 Scalar(0),
                 T(50, <<1, 2>>, <<3, 2>>, <<1, 3>>)),
  \* extent 3, A = {0,-2,-4} = [-4,1), B = [3,6), C at 1 touches A.
  touch3   |-> D(T(0, <<1>>, <<3>>, <<-2>>), T(3, <<1>>, <<3>>, <<1>>), Scalar(1)),
  \* nonzero A base (2) different from B's (3): A read at 2, 3.
  abase    |-> D(T(2, <<1>>, <<2>>, <<1>>), T(3, <<1>>, <<2>>, <<1>>), Scalar(0))
]
ASSUME MustCover \subseteq DOMAIN Cov
ASSUME \A c \in MustCover : InSpace(Cov[c]) /\ Status(Cov[c]) = "SUCCESS"

\* Pins on coverage descriptors (design models only):
\*  - fusion HAPPENS when on its boundary (a lowering that never fuses is
\*    correct but violates step 8; FusionPreservesAddresses catches the
\*    opposite error, fusing off the boundary);
\*  - the lowering offsets A by A's base (and B by B's).
ASSUME Drop = "none" =>
         LET L3 == Lower(Cov.fuse3)  LN == Lower(Cov.fuseneg)  LO == Lower(Cov.fuseoff)
         IN  /\ L3.fused["I"] = <<[e |-> 6, s |-> [A |-> 1, B |-> 0, C |-> 1]]>>
             /\ LN.fused["K"] = <<[e |-> 4, s |-> [A |-> -1, B |-> 0, C |-> 0]]>>
             /\ Len(LO.fused["K"]) = 2
ASSUME Drop = "none" =>
         /\ Lower(Cov.abase).readsA = {2, 3} /\ Lower(Cov.abase).readsB = {3, 4}
         /\ Lower(Cov.abase).contrib = Einsum(Cov.abase).bag
=============================================================================
