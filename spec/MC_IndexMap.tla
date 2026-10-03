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
(*  pos_swap     role swap (§5 step 8): 3 labels, ranks <= 2, extents      *)
(*               {1,2,3}, C strides {-1,0,1,2,3}: swapped / unswapped,     *)
(*               fastest mode in L, the extent > 1 filter ON its boundary  *)
(*               (an extent-1 I mode of stride 0), swap at extent 3        *)
(*               (m, n = 3, 2 -> 2, 3, nh = 4). Every claim of the module  *)
(*               also runs on the swapped lowering in every model above.   *)
(*  pos_swap_order  4 labels, extent 2, operand strides with zeros: the    *)
(*               orders on the POST-swap roles (K's big operand at m = n), *)
(*               position from the caller's label order (full K ties).     *)
(*  pos_limit_n  GemmMax = 3 (real 2^31 - 1), PadQ = 2, extents {0,2,3},   *)
(*               EMULATE and DIRECT: m = 3 ON the boundary; n = 3 <=       *)
(*               GemmMax but nh = 4 rejected; the post-swap nh decides;    *)
(*               DIRECT, an empty output, k = 0 and K-empty are exempt.   *)
(*  pos_limit_m  GemmMax = 2: m = 2 ON, m = 3 rejected (EMULATE only).     *)
(*  pos_kappa    §3.2 accuracy policy kappa = ceil(sqrt(k)) >= T ->        *)
(*               NOT_SUPPORTED, symbolic T in {1,2,3} (real T(N) >= 180),  *)
(*               extents {0,2,3}, GemmMax = 100 (out of the way). Execs:   *)
(*               EMULATE+RN at T = 1, 2, 3; DIRECT+RN and EMULATE+TRUNC at *)
(*               T = 1 (every k >= 2 has kappa >= T there, so the scope    *)
(*               exemptions are exercised maximally). kappa = T - 1        *)
(*               accepted (k = 4, T = 3) and kappa = T rejected (k = 6,    *)
(*               T = 3; k = 2, T = 2: the k >= 2 boundary); exempt ON each *)
(*               scope boundary: DIRECT, TRUNC, an empty output, K-empty   *)
(*               k = 1 at T = 1 (kappa = T), k = 0.                        *)
(*  pos_contract_ws     imd_contract, AlignWS = 16: workspace placed       *)
(*               touching / one byte into C, A, B, ws_bytes = size - 1,   *)
(*               size, size + 8 (slack overlapping A is fine), both modes. *)
(*  pos_contract_order  null x misalignment x ws placement x ws_bytes:    *)
(*               every pair of §6 checks firing together; the null rule's  *)
(*               conditions ON their boundaries (C always; A/B at k = 0    *)
(*               and alpha = 0; ws at workspace_size = 0).                 *)
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
(*  neg_noswap / neg_swap_filter   SwapMakesJFastest: no swap / the swap   *)
(*               decided over all C modes (swape1 is swapped).            *)
(*  neg_swap_late_ws / neg_swap_late_limit   swap after sizing: a region   *)
(*               overruns (WorkspaceFits) / a GEMM with nh > GemmMax is    *)
(*               launched (GemmDimsFit).                                   *)
(*  neg_order_pre_swap  orders before the swap: CanonicalOrderSorted.     *)
(*  neg_pos_swapped     position from the swapped descriptor:              *)
(*               CanonicalOrderExact.                                      *)
(*  neg_limit_drop / neg_limit_n   GemmDimsFit: no limit / n checked, not  *)
(*               nh.  neg_limit_direct / _empty / _k   LimitScope: limit   *)
(*               applied in DIRECT / to an empty output / with k < 2.      *)
(*  neg_kappa_drop  Drop "Kappa": an RN GEMM with kappa >= T is accepted:  *)
(*               KappaAccepted.                                            *)
(*  neg_kappa_direct / _empty / _k / _trunc   KappaScope: the policy also  *)
(*               applied in DIRECT / to an empty output / with k < 2 (K-   *)
(*               empty k = 1 at T = 1) / with TRUNC. Each scope condition  *)
(*               is necessary.                                             *)
(*  neg_null, neg_ws_alias, neg_ws_range, neg_misaligned,                  *)
(*  neg_insufficient   one §6 check dropped (ws_range: [ws, ws + ws_bytes)*)
(*               instead of [ws, ws + workspace_size)): NoNullDeref,      *)
(*               WsNoAlias, AliasingSound, AlignedAccess, WsSufficient.    *)
(*  neg_null_strict  any null rejected (A/B at alpha = 0 or k = 0, ws at  *)
(*               workspace_size = 0): StatusCorrect (over-rejection).      *)
(*  neg_null_ws_lax  a null ws accepted although workspace_size > 0:      *)
(*               NoNullDeref.                                              *)
(*  neg_order_*  the §6 checks reordered (null/empty, empty last,          *)
(*               MISALIGNED before ALIASING, INSUFFICIENT before           *)
(*               MISALIGNED): StatusCorrect. These orders are POLICY (no  *)
(*               semantic invariant depends on them); the models show the  *)
(*               enumerated space distinguishes every adjacent pair.       *)
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
\* Swap models: C strides with 0 (an extent-1 mode may have it), signs,
\* and 3 (dense at extent 3).
SwS(r)    == [1..r -> {-1, 0, 1, 2, 3}]
\* Swap -> order model: operand strides with zeros (full K-key ties) and
\* both mode orders.
Ord2(r)   == IF r = 2 THEN {<<1, 0>>, <<0, 1>>, <<1, 2>>, <<2, 1>>} ELSE [1..r -> {0, 1}]
\* Contract models: A at elements {0,1} (bytes [0,16)), B at {2,3} ([16,32)),
\* C at 1 (aliases A), 4 (touches B) or 5.
BC145     == {1, 4, 5}
B2        == {2}
Pm1(r)    == [1..r -> {-1, 1}]

-----------------------------------------------------------------------------
(* Execute-time argument sets (see IndexMap XDef).                        *)

XDefSet   == {XDef}
XDirect   == [XDef EXCEPT !.mode = "DIRECT"]
ExModes   == {XDef, XDirect}
Mis0      == [X \in Tens |-> 0]
MisA4     == [Mis0 EXCEPT !["A"] = 4]
MisC4     == [Mis0 EXCEPT !["C"] = 4]
\* Workspace placements: far; touching (delta 0) and one byte over
\* (delta -1 after / +1 before) at C and at A; one byte into B.
WsPlaces  == {<<"far", "after", 0>>, <<"C", "after", 0>>, <<"C", "after", -1>>,
              <<"C", "before", 0>>, <<"C", "before", 1>>, <<"A", "after", 0>>,
              <<"A", "before", 0>>, <<"A", "before", 1>>, <<"B", "after", -1>>}
Ex(md, nl, ms, wp, dw) == [XDef EXCEPT !.mode = md, !.nul = nl, !.mis = ms, !.wsp = wp, !.dws = dw]
ExA0(md, nl, ms, wp, dw) == [Ex(md, nl, ms, wp, dw) EXCEPT !.a0 = TRUE]
\* ws placement x insufficient/exact/larger ws_bytes x both modes.
ExWs      == {Ex(md, {}, Mis0, wp, dw) : md \in {"EMULATE", "DIRECT"},
                                          wp \in WsPlaces, dw \in {-1, 0, 8}}
\* null x misalignment x (far / touching / overlapping ws) x ws_bytes,
\* plus alpha = 0 with a null A or B (the §6 null rule's alpha condition).
ExOrder   == {Ex("EMULATE", nl, ms, wp, dw) :
                nl \in {{}, {"A"}, {"B"}, {"C"}, {"ws"}}, ms \in {Mis0, MisA4, MisC4},
                wp \in {<<"far", "after", 0>>, <<"C", "after", 0>>, <<"C", "after", -1>>},
                dw \in {-1, 0}}
             \cup {ExA0("EMULATE", nl, Mis0, <<"far", "after", 0>>, 0) :
                    nl \in {{}, {"A"}, {"B"}, {"A", "B"}, {"ws"}}}
\* §3.2 kappa policy: mode x rounding (rn) x symbolic T (tn).
ExK(md, r, t) == [XDef EXCEPT !.mode = md, !.rn = r, !.tn = t]
ExKappa   == {ExK("EMULATE", TRUE, t) : t \in 1..3}
             \cup {ExK("DIRECT", TRUE, 1), ExK("EMULATE", FALSE, 1)}

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
ASSUME Drop = "Fusion" => LowerTop(WFuse, "EMULATE").contrib # Einsum(WFuse).bag
\* The same descriptor without step 6: label 2 is not summed.
ASSUME Drop = "Single" => LowerTop(WFuse, "EMULATE").contrib # Einsum(WFuse).bag
ASSUME Drop = "none"   => LowerTop(WFuse, "EMULATE").contrib = Einsum(WFuse).bag

\* C[1,2] with strides (1,1): |s_2| = span = 1, one below the boundary.
WCStr == D(T(0, <<1>>, <<2>>, <<1>>), T(0, <<2>>, <<2>>, <<1>>),
           T(100, <<1, 2>>, <<2, 2>>, <<1, 1>>))
ASSUME Drop # "CStride" => Status(WCStr) = "INVALID_VALUE"
ASSUME Drop = "CStride" => \E a \in DOMAIN LowerTop(WCStr, "EMULATE").writes : LowerTop(WCStr, "EMULATE").writes[a] = 2

\* C[1,1] (repeated C label), strides (1,2): only the diagonal is written.
WRepC == D(T(0, <<1>>, <<2>>, <<1>>), T(0, <<1>>, <<2>>, <<1>>),
           T(100, <<1, 1>>, <<2, 2>>, <<1, 2>>))
ASSUME Drop # "RepeatC" => Status(WRepC) = "NOT_SUPPORTED"
ASSUME Drop = "RepeatC" => DOMAIN LowerTop(WRepC, "EMULATE").writes = {100, 103}
                           /\ Elems(WRepC, "C") = {100, 101, 102, 103}

\* Label 1 has extent 1 in A, 2 in B: lowered with 2, A is read at 1.
WExt == D(T(0, <<1>>, <<1>>, <<1>>), T(0, <<1>>, <<2>>, <<1>>), Scalar(100))
ASSUME Drop # "Extents" => Status(WExt) = "INVALID_VALUE"
ASSUME Drop = "Extents" => 1 \in LowerTop(WExt, "EMULATE").readsA /\ Elems(WExt, "A") = {0}

\* C (scalar at 1) is element 1 of A = [1] (stride 1, extent 2).
WAl == D(T(0, <<1>>, <<2>>, <<1>>), T(50, <<1>>, <<2>>, <<1>>), Scalar(1))
ASSUME Drop # "Aliasing" => Status(WAl) = "ALIASING"
ASSUME Drop = "Aliasing" => Status(WAl) = "SUCCESS" /\ 1 \in Elems(WAl, "A")

\* C-only label 2 (broadcast) of extent 2: rejected in v1; with the rule
\* dropped, the lowering (I, stride 0 in A) is exact.
WCOnly == D(T(0, <<1>>, <<2>>, <<1>>), T(0, <<1>>, <<2>>, <<1>>),
            T(100, <<2>>, <<2>>, <<1>>))
ASSUME Drop # "COnly" => Status(WCOnly) = "NOT_SUPPORTED"
ASSUME Drop = "COnly" => /\ LowerTop(WCOnly, "EMULATE").contrib = Einsum(WCOnly).bag
                         /\ DOMAIN LowerTop(WCOnly, "EMULATE").writes = {100, 101}

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
  \* of extent 3 gives k = 3, kh = 4 (PadQ 2); C's fastest mode is in I,
  \* so the roles swap: m = 2, n = 3, nh = 4 (padded).
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
  abase    |-> D(T(2, <<1>>, <<2>>, <<1>>), T(3, <<1>>, <<2>>, <<1>>), Scalar(0)),
  \* role swap: C[i,j] with C's fastest mode i (stride 1) in I -> swapped.
  swapI    |-> D(T(0, <<1, 2>>, <<2, 2>>, <<1, 2>>), T(0, <<2, 3>>, <<2, 2>>, <<1, 2>>),
                 T(100, <<1, 3>>, <<2, 2>>, <<1, 2>>)),
  \* the same with C's fastest mode j in J: not swapped.
  noswapJ  |-> D(T(0, <<1, 2>>, <<2, 2>>, <<1, 2>>), T(0, <<2, 3>>, <<2, 2>>, <<1, 2>>),
                 T(100, <<1, 3>>, <<2, 2>>, <<2, 1>>)),
  \* C's fastest mode is an L label (in A, B and C): not swapped.
  fastL    |-> D(T(0, <<1, 2>>, <<2, 2>>, <<1, 2>>), T(0, <<1, 3>>, <<2, 2>>, <<1, 2>>),
                 T(100, <<1, 2>>, <<2, 2>>, <<1, 2>>)),
  \* the extent > 1 filter ON its boundary: the I mode has extent 1 and
  \* stride 0 (smaller than j's 1) and must be ignored: not swapped.
  swape1   |-> D(T(0, <<1, 2>>, <<1, 2>>, <<1, 2>>), T(0, <<2, 3>>, <<2, 2>>, <<1, 2>>),
                 T(100, <<1, 3>>, <<1, 2>>, <<0, 1>>)),
  \* swap at extent 3: user m = 3, n = 2 become m = 2, n = 3, nh = 4.
  swap3    |-> D(T(0, <<1, 2>>, <<3, 2>>, <<1, 2>>), T(0, <<2, 3>>, <<2, 2>>, <<1, 2>>),
                 T(100, <<1, 3>>, <<3, 2>>, <<1, 3>>)),
  \* order after the swap, m = n = 2: K = {2 (A-only, stride 2 in A),
  \* 4 (B-only, stride 2 in B)}. Post-swap big = A' = the caller's B, so
  \* K = <<2, 4>> (label 2 has stride 0 there); ordering BEFORE the swap
  \* (big = the caller's A) gives <<4, 2>>.
  ordK     |-> D(T(0, <<1, 2>>, <<2, 2>>, <<1, 2>>), T(0, <<3, 4>>, <<2, 2>>, <<1, 2>>),
                 T(100, <<1, 3>>, <<2, 2>>, <<1, 2>>)),
  \* full tie in K (strides 0 in both operands): position decides, from the
  \* CALLER's A.l ++ B.l ++ C.l: <<2, 4>>; the swapped descriptor's
  \* B.l ++ A.l ++ C.l would give <<4, 2>>.
  ordPos   |-> D(T(0, <<1, 2>>, <<2, 2>>, <<1, 0>>), T(0, <<3, 4>>, <<2, 2>>, <<1, 0>>),
                 T(100, <<1, 3>>, <<2, 2>>, <<1, 2>>))
]

(* Coverage with execute arguments and an EXPECTED status (rejections     *)
(* included: the boundary + 1 cases): <<descriptor, Execs record, status>>.*)
\* GEMM limits (pos_limit_n: GemmMax = 3, PadQ = 2). C at 100.
LA(e1, e2)     == T(0, <<1, 2>>, <<e1, e2>>, <<1, 2>>)
LB(e2, e3)     == T(0, <<2, 3>>, <<e2, e3>>, <<1, 2>>)
LC(e1, e3, s)  == T(100, <<1, 3>>, <<e1, e3>>, s)
\* m = 3 ON the boundary, n = 2 = nh; C's fastest mode j: no swap.
DLm3   == D(LA(3, 2), LB(2, 2), LC(3, 2, <<2, 1>>))
\* n = 3 <= GemmMax but nh = 4 > GemmMax (the padding matters).
DLn3   == D(LA(2, 2), LB(2, 3), LC(2, 3, <<3, 1>>))
\* user m = 3, n = 2 pass the limits, but C's fastest mode is i: after the
\* swap m = 2, n = 3, nh = 4 > GemmMax.
DLswap == D(LA(3, 2), LB(2, 2), LC(3, 2, <<1, 3>>))
\* nh = 4 but the output is empty (m = 0).
DLempty == D(LA(0, 2), LB(2, 3), LC(0, 3, <<3, 1>>))
\* K-empty (outer product, dispatched to DIRECT) with nh = 4.
DLkempty == D(T(0, <<1>>, <<2>>, <<1>>), T(0, <<2>>, <<3>>, <<1>>),
              T(100, <<1, 2>>, <<2, 3>>, <<3, 1>>))
\* k = 0 (C <- beta*C, no GEMM) with nh = 4.
DLk0   == D(LA(2, 0), LB(0, 3), LC(2, 3, <<3, 1>>))
\* pos_limit_m (GemmMax = 2): m = 2 ON the boundary.
DLm2   == D(LA(2, 2), LB(2, 2), LC(2, 2, <<2, 1>>))

\* §3.2 kappa policy (pos_kappa). Dot products C = sum A B (m = n = 1):
\* one K label of extent e (k = e) or two of extents e1, e2 (k = e1*e2).
DK1(e)       == D(T(0, <<1>>, <<e>>, <<1>>), T(0, <<1>>, <<e>>, <<1>>), Scalar(100))
DK2(e1, e2)  == D(T(0, <<1, 2>>, <<e1, e2>>, <<1, 2>>), T(0, <<1, 2>>, <<e1, e2>>, <<1, 2>>),
                  Scalar(100))
\* empty output (m = 0) with k = 3 (kappa = 2).
DKempty      == D(T(0, <<1, 2>>, <<0, 3>>, <<1, 2>>), T(0, <<2>>, <<3>>, <<1>>),
                  T(100, <<1>>, <<0>>, <<1>>))

\* Contract models: A = [0,2) elements, B = [2,4), C a scalar.
DDot(c)  == D(T(0, <<1>>, <<2>>, <<1>>), T(2, <<1>>, <<2>>, <<1>>), Scalar(c))
\* empty output (m = 0); k = 0 (SCALE).
DEmpty   == D(T(0, <<1>>, <<0>>, <<1>>), Scalar(2), T(5, <<1>>, <<0>>, <<1>>))
DK0      == D(T(0, <<1>>, <<0>>, <<1>>), T(2, <<1>>, <<0>>, <<1>>), Scalar(5))
XE(nl, ms, wp, dw) == Ex("EMULATE", nl, ms, wp, dw)

CovX == [
  \* --- GEMM limits, GemmMax = 3 ---
  limm3    |-> <<DLm3, XDef, "SUCCESS">>,
  limn3    |-> <<DLn3, XDef, "INVALID_VALUE">>,
  limdir   |-> <<DLn3, XDirect, "SUCCESS">>,
  limswap  |-> <<DLswap, XDef, "INVALID_VALUE">>,
  limempty |-> <<DLempty, XDef, "SUCCESS">>,
  \* k < 2 runs no GEMM (K-empty -> DIRECT; k = 0 -> beta*C): not limited
  \* (the k >= 2 boundary of row 5).
  limkempty |-> <<DLkempty, XDef, "SUCCESS">>,
  limk0    |-> <<DLk0, XDef, "SUCCESS">>,
  \* --- GEMM limits, GemmMax = 2 ---
  limm2    |-> <<DLm2, XDef, "SUCCESS">>,
  limm3rej |-> <<DLm3, XDef, "INVALID_VALUE">>,
  limm3dir |-> <<DLm3, XDirect, "SUCCESS">>,
  \* --- §3.2 kappa >= T policy (pos_kappa, GemmMax = 100) ---
  \* kappa = T - 1 accepted, kappa = T rejected (k = 4: kappa 2; k = 6: 3).
  kapk4t3  |-> <<DK2(2, 2), ExK("EMULATE", TRUE, 3), "SUCCESS">>,
  kapk6t3  |-> <<DK2(2, 3), ExK("EMULATE", TRUE, 3), "NOT_SUPPORTED">>,
  \* the smallest k that runs a GEMM (k = 2, kappa = 2 = T) is rejected.
  kapk2t2  |-> <<DK1(2), ExK("EMULATE", TRUE, 2), "NOT_SUPPORTED">>,
  \* exempt, each ON its scope boundary with kappa >= T:
  kapdir   |-> <<DK2(2, 3), ExK("DIRECT", TRUE, 1), "SUCCESS">>,
  kaptrunc |-> <<DK2(2, 3), ExK("EMULATE", FALSE, 1), "SUCCESS">>,
  kapempty |-> <<DKempty, ExK("EMULATE", TRUE, 2), "SUCCESS">>,
  \* K-empty (k = 1, kappa = 1 = T at T = 1): DIRECT by descriptor.
  kapk1t1  |-> <<DLkempty, ExK("EMULATE", TRUE, 1), "SUCCESS">>,
  kapk0t1  |-> <<DLk0, ExK("EMULATE", TRUE, 1), "SUCCESS">>,
  \* --- workspace placement / size (pos_contract_ws, AlignWS = 16) ---
  ctouchB  |-> <<DDot(4), XDef, "SUCCESS">>,                    \* C [32,40) touches B
  caliasA  |-> <<DDot(1), XDef, "ALIASING">>,
  wsafter0 |-> <<DDot(5), XE({}, Mis0, <<"C", "after", 0>>, 0), "SUCCESS">>,
  wsafter1 |-> <<DDot(5), XE({}, Mis0, <<"C", "after", -1>>, 0), "ALIASING">>,
  wsbefA0  |-> <<DDot(5), XE({}, Mis0, <<"A", "before", 0>>, 0), "SUCCESS">>,
  wsbefA1  |-> <<DDot(5), XE({}, Mis0, <<"A", "before", 1>>, 0), "ALIASING">>,
  wsintoB  |-> <<DDot(5), XE({}, Mis0, <<"B", "after", -1>>, 0), "ALIASING">>,
  wsmis    |-> <<DDot(4), XE({}, Mis0, <<"C", "after", 0>>, 0), "MISALIGNED">>,
  wsshort  |-> <<DDot(5), XE({}, Mis0, <<"far", "after", 0>>, -1), "INSUFFICIENT_WORKSPACE">>,
  \* ws_bytes > workspace_size and the slack overlaps A: still SUCCESS.
  wsslack  |-> <<DDot(5), XE({}, Mis0, <<"A", "before", 0>>, 8), "SUCCESS">>,
  wsdirect |-> <<DDot(5), Ex("DIRECT", {}, Mis0, <<"C", "after", 0>>, 0), "SUCCESS">>,
  \* --- status order (pos_contract_order) ---
  \* §6 1 (DESIGN after M1): a null C beats the empty output; a null ws is
  \* fine when workspace_size = 0 (empty output) and rejected when > 0; a
  \* null A or B is fine when k = 0 or alpha = 0 (never read), rejected
  \* otherwise. Each is ON the boundary of its condition.
  nullCempty  |-> <<DEmpty, XE({"C"}, Mis0, <<"far", "after", 0>>, 0), "INVALID_VALUE">>,
  nullwsempty |-> <<DEmpty, XE({"ws"}, Mis0, <<"far", "after", 0>>, 0), "SUCCESS">>,
  nullws   |-> <<DDot(5), XE({"ws"}, Mis0, <<"far", "after", 0>>, 0), "INVALID_VALUE">>,
  nullAk0  |-> <<DK0, XE({"A"}, Mis0, <<"far", "after", 0>>, 0), "SUCCESS">>,
  nullA    |-> <<DDot(5), XE({"A"}, Mis0, <<"far", "after", 0>>, 0), "INVALID_VALUE">>,
  nullB    |-> <<DDot(5), XE({"B"}, Mis0, <<"far", "after", 0>>, 0), "INVALID_VALUE">>,
  nullABa0 |-> <<DDot(5), ExA0("EMULATE", {"A", "B"}, Mis0, <<"far", "after", 0>>, 0), "SUCCESS">>,
  nullwsa0 |-> <<DDot(5), ExA0("EMULATE", {"ws"}, Mis0, <<"far", "after", 0>>, 0), "INVALID_VALUE">>,
  emptymis |-> <<DEmpty, XE({}, MisC4, <<"C", "after", -1>>, -1), "SUCCESS">>,
  aliasmis |-> <<DDot(1), XE({}, MisC4, <<"far", "after", 0>>, 0), "ALIASING">>,
  wsamis   |-> <<DDot(5), XE({}, Mis0, <<"C", "after", -1>>, -1), "ALIASING">>,
  misins   |-> <<DDot(5), XE({}, MisA4, <<"far", "after", 0>>, -1), "MISALIGNED">>,
  nullalias |-> <<DDot(1), XE({"C"}, Mis0, <<"far", "after", 0>>, 0), "INVALID_VALUE">>,
  ordok    |-> <<DDot(5), XE({}, Mis0, <<"far", "after", 0>>, 0), "SUCCESS">>
]
ASSUME MustCover \subseteq DOMAIN Cov \cup DOMAIN CovX
\* The null-ws boundary: an empty output needs no workspace, a dot does.
ASSUME WsSizeDesign(DEmpty, "EMULATE") = 0 /\ WsSizeDesign(DDot(5), "EMULATE") > 0
ASSUME \A c \in MustCover \cap DOMAIN Cov : InSpace(Cov[c]) /\ Status(Cov[c]) = "SUCCESS"
ASSUME \A c \in MustCover \cap DOMAIN CovX :
          /\ InSpace(CovX[c][1]) /\ CovX[c][2] \in Execs
          /\ FullStatus(CovX[c][1], CovX[c][2]) = CovX[c][3]
          /\ ExpectedStatus(CovX[c][1], CovX[c][2]) = CovX[c][3]

\* Pins on the swap coverage (design models only).
ASSUME Drop = "none" =>
         /\ LowerTop(Cov.swapI, "EMULATE").swap
         /\ ~LowerTop(Cov.noswapJ, "EMULATE").swap
         /\ ~LowerTop(Cov.fastL, "EMULATE").swap
         /\ ~LowerTop(Cov.swape1, "EMULATE").swap
         /\ LET L == LowerTop(Cov.swap3, "EMULATE")
            IN  L.swap /\ L.m = 2 /\ L.n = 3 /\ L.nh = 4
             /\ UDims(Cov.swap3).m = 3 /\ UDims(Cov.swap3).n = 2
\* SwapFilter's witness: it swaps swape1 (the extent-1 I mode, stride 0).
ASSUME Drop = "SwapFilter" => LowerTop(Cov.swape1, "EMULATE").swap
\* Swap -> order pins (both descriptors are swapped).
ASSUME Drop = "none" =>
         /\ LowerTop(Cov.ordK, "EMULATE").swap /\ LowerTop(Cov.ordPos, "EMULATE").swap
         /\ LowerTop(Cov.ordK, "EMULATE").ord["K"] = <<2, 4>>
         /\ LowerTop(Cov.ordPos, "EMULATE").ord["K"] = <<2, 4>>
ASSUME Drop = "OrderPreSwap" => LowerTop(Cov.ordK, "EMULATE").ord["K"] = <<4, 2>>
ASSUME Drop = "PosSwapped"   => LowerTop(Cov.ordPos, "EMULATE").ord["K"] = <<4, 2>>

\* Pins on coverage descriptors (design models only):
\*  - fusion HAPPENS when on its boundary (a lowering that never fuses is
\*    correct but violates step 8; FusionPreservesAddresses catches the
\*    opposite error, fusing off the boundary);
\*  - the lowering offsets A by A's base (and B by B's).
ASSUME Drop = "none" =>
         LET L3 == LowerTop(Cov.fuse3, "EMULATE")  LN == LowerTop(Cov.fuseneg, "EMULATE")  LO == LowerTop(Cov.fuseoff, "EMULATE")
         \* fuse3's C has its fastest mode (label 1, stride 1) in I, so the roles
         \* are swapped: the fused group is J, A (scalar) is in role B.
         IN  /\ L3.swap
             /\ L3.fused["J"] = <<[e |-> 6, s |-> [A |-> 0, B |-> 1, C |-> 1]]>>
             /\ LN.fused["K"] = <<[e |-> 4, s |-> [A |-> -1, B |-> 0, C |-> 0]]>>
             /\ Len(LO.fused["K"]) = 2
ASSUME Drop = "none" =>
         /\ LowerTop(Cov.abase, "EMULATE").readsA = {2, 3} /\ LowerTop(Cov.abase, "EMULATE").readsB = {3, 4}
         /\ LowerTop(Cov.abase, "EMULATE").contrib = Einsum(Cov.abase).bag
=============================================================================
