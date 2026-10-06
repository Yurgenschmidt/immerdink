-------------------------- MODULE MC_PackedOperand --------------------------
(***************************************************************************)
(* Model values for PackedOperand (DESIGN §6b). Each cfg substitutes        *)
(* Cmp and Strm with one of the scenarios below (cfgs cannot hold records  *)
(* or negative numbers).                                                   *)
(*                                                                         *)
(* Scaling (real -> model): moduli 256, 255, 253, ... -> 8, 7, 5, 3 (T = 5 *)
(* at N = 2, 11 at N = 3, 20 at N = 4; κ = ⌈√4⌉ = 2 < T, so RN accepts);   *)
(* k̂ = ⌈k⌉₁₆ -> ⌈k⌉₈ (k = 4: four padding positions); n̂ = ⌈n⌉₁₆ -> ⌈n⌉₂ *)
(* (r = 1: one overread row in the B role); one row of X, one column of Y, *)
(* b = 1. The values are chosen so σ < 0 occurs (S up to 36 > 16 = the RN *)
(* threshold at N = 2), with odd entries, so RN and TRUNC quantize         *)
(* differently, and so the two K orders pair different entries (E1 = E2 = *)
(* 2, both orders present).                                                *)
(*                                                                         *)
(* CompatSpec. Positive models (must pass; Cmp.check = FALSE: the pack is  *)
(* built to the plan's key; muts = {"none"} unless noted):                 *)
(*   pos_rn2       N = 2, RN, roles A and B, g ∈ {1, 2}, v1 default K      *)
(*                 order "21" ≠ the pack's "12" (adoption matters)         *)
(*   pos_trunc2    as pos_rn2 with TRUNC                                   *)
(*   pos_rn3       N = 3, g ∈ {1, 2, 3} (a short last group at g = 2)      *)
(*   pos_both      both operands packed (the partner's pack in the other   *)
(*                 role), N = 2                                            *)
(*   pos_flags     X has non-finite entries: flagged outputs take the      *)
(*                 native path on the FP64 data in the adopted order       *)
(*   pos_key_check plan creation compares keys; X's pack differs in one    *)
(*                 key component (N, rnd, kset, kalign) or the plan does   *)
(*                 not adopt the order: every one is rejected              *)
(*                 (PACK_MISMATCH), the matching pack is accepted          *)
(*   pos_key_check_both  both packed, the partner's pack in the other K    *)
(*                 order: rejected                                         *)
(* Negative models (must fail; one mutation, no key check):                *)
(*   neg_order          CEqual     plan keeps v1's default K order         *)
(*   neg_n              CEqual     pack made with N + 1                    *)
(*   neg_rnd            CEqual     pack made with the other rounding       *)
(*   neg_kset           CEqual     pack's K set lacks a partner-only label *)
(*   neg_kalign         CEqual     pack's k̂ alignment (format) differs     *)
(*   neg_partner_order  CEqual     both packed, K orders differ            *)
(*   neg_brole          CEqual     B role read with v1 RB block strides    *)
(*   neg_notail         NoOOBRead  no zeroed tail: the B-role GEMM's       *)
(*                                 padded rows of the last block overrun   *)
(*   neg_nopad          WEqual     both packs leave K padding junk         *)
(*   neg_noflags        CEqual     pack stores no non-finite flags         *)
(* Witnesses (must fail: the positives are not vacuous):                   *)
(*   neg_witness_mismatch      NeverMismatch    (pos_key_check rejects)    *)
(*   neg_witness_native_order  NativeOrderFree  (pos_flags: a flagged      *)
(*                 output depends on the K order, so a packed contraction  *)
(*                 equals the unpacked one only in the ADOPTED order; for  *)
(*                 unflagged outputs EmulatedOrderFree shows the order is  *)
(*                 irrelevant)                                             *)
(*                                                                         *)
(* StreamSpec. Positive (must pass):                                       *)
(*   pos_stream_same    pack, both contracts and a re-pack on ONE stream   *)
(*   pos_stream_events  contracts on streams 2 and 3 (concurrent readers), *)
(*                      RAW event pack -> contracts, WAR events contracts  *)
(*                      -> re-pack on stream 1                             *)
(*   pos_stream_free    as pos_stream_events, the buffer freed instead     *)
(*   pos_stream_writex  as pos_stream_events, the FP64 X overwritten       *)
(* Negative (NoStaleRead):                                                 *)
(*   neg_stream_no_raw       contract on another stream, no RAW event      *)
(*   neg_stream_no_war       re-pack without waiting for the readers       *)
(*   neg_stream_free_early   free without waiting for the readers          *)
(*   neg_stream_writex_early FP64 X overwritten while a contract may still *)
(*                           read it (K5's native path)                    *)
(* Witness: neg_witness_concurrent  NeverConcurrentReads (pos_stream_events)*)
(***************************************************************************)
EXTENDS PackedOperand

VXa   == {-3, 1, 2}
VYa   == {3, -2, 1}
VXinf == {-3, 2, INF}

CmpBase == [N |-> 2, rnd |-> "RN", roles |-> {"A", "B"}, gs |-> {1, 2},
            defOrder |-> "21", both |-> FALSE, muts |-> {"none"},
            check |-> FALSE, vx |-> VXa, vy |-> VYa]
CmpTrunc2   == [CmpBase EXCEPT !.rnd = "TRUNC"]
CmpRN3      == [CmpBase EXCEPT !.N = 3, !.gs = {1, 2, 3}]
CmpBoth     == [CmpBase EXCEPT !.both = TRUE]
CmpFlags    == [CmpBase EXCEPT !.vx = VXinf]
CmpKeyCheck == [CmpBase EXCEPT !.check = TRUE,
                    !.muts = {"none", "N", "rnd", "kset", "kalign", "order"}]
CmpKeyCheckBoth == [CmpBoth EXCEPT !.check = TRUE, !.muts = {"none", "partnerOrder"}]
\* negatives: one mutation each
CmpOrder    == [CmpBase EXCEPT !.muts = {"order"}]
CmpN        == [CmpBase EXCEPT !.muts = {"N"}]
CmpRnd      == [CmpBase EXCEPT !.muts = {"rnd"}]
CmpKset     == [CmpBase EXCEPT !.muts = {"kset"}]
CmpKalign   == [CmpBase EXCEPT !.muts = {"kalign"}]
CmpPartnerOrder == [CmpBoth EXCEPT !.muts = {"partnerOrder"}]
CmpBrole    == [CmpBase EXCEPT !.roles = {"B"}, !.muts = {"brole"}]
CmpNotail   == [CmpBase EXCEPT !.roles = {"B"}, !.muts = {"notail"}]
CmpNopad    == [CmpBoth EXCEPT !.muts = {"nopad"}]
CmpNoflags  == [CmpFlags EXCEPT !.muts = {"noflags"}]

\* streams: pack, c1, c2, the later write (post) on postSt
StrmSame    == [pack |-> 1, c1 |-> 1, c2 |-> 1, postSt |-> 1, post |-> "repack",
                raw |-> FALSE, war |-> FALSE]
StrmEvents  == [pack |-> 1, c1 |-> 2, c2 |-> 3, postSt |-> 1, post |-> "repack",
                raw |-> TRUE, war |-> TRUE]
StrmFree    == [StrmEvents EXCEPT !.post = "free"]
StrmWriteX  == [StrmEvents EXCEPT !.post = "writeX"]
StrmNoRaw   == [StrmEvents EXCEPT !.raw = FALSE]
StrmNoWar   == [StrmEvents EXCEPT !.war = FALSE]
StrmFreeEarly  == [StrmFree EXCEPT !.war = FALSE]
StrmWriteXEarly == [StrmWriteX EXCEPT !.war = FALSE]
=============================================================================
