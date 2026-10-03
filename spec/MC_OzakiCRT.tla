---------------------------- MODULE MC_OzakiCRT ----------------------------
(***************************************************************************)
(* Model values for OzakiCRT (.cfg files cannot hold negative literals or  *)
(* tuples, so they live here and the cfgs use `Const <- Def`).            *)
(*                                                                         *)
(* Scaled moduli = prefixes of the greedy pairwise-coprime table from      *)
(* 2^ResBits down, the rule that gives DESIGN §2's 256, 255, 253, ...:     *)
(*   ResBits 2: <<4,3>>            P = 12      H = 5                        *)
(*   ResBits 3: <<8,7,5,3>>        P = 840     H = 419                      *)
(*              prefix <<8,7,5>>   P = 280     H = 139                      *)
(*   ResBits 4: <<16,15,13,11,7>>  P = 240240  H = 120119                   *)
(* NOT a greedy table (negative models only): <<9,8,7,5>> P = 2520, H =   *)
(* 1259, ResBits 3: p = 9 = 2^3 + 1 is the boundary+1 of PreResidue (the   *)
(* real-width analogue: a table starting at 257 = 2^8 + 1, prime, coprime  *)
(* to everything, but SMod range -128..128 and 128 is not an int8).        *)
(* Accumulator widths are chosen so that KCMax = AccMax \div 4^(ResBits-1) *)
(* has the real shape 2^j - 1 (real: 2^17 - 1 = 131071):                   *)
(*   ResBits 2, AccBits 5: AccMax 15,  KCMax 3                              *)
(*   ResBits 3, AccBits 6: AccMax 31,  KCMax 1                              *)
(*   ResBits 3, AccBits 7: AccMax 63,  KCMax 3                              *)
(*   ResBits 4, AccBits 8: AccMax 127, KCMax 1                              *)
(*                                                                         *)
(* POSITIVE models (must pass; each ON its boundary, enforced by the       *)
(* spec's Coverage ASSUME):                                                *)
(*  pos_crt_r3_k1  <<8,7,5,3>>, K=1, a in -420..420, b in -1..1. a covers  *)
(*                 all 840 residue classes, so every W tuple is reached;   *)
(*                 SumAbs reaches H = 419 and H+1 = 420 (both sides of the *)
(*                 CRTExact guard).                                        *)
(*  pos_crt_r3_k2  <<8,7,5>>, K=2, KC=1=KCMax (two chunks: carry), a in    *)
(*                 -12..12, b in -7..7: SumAbs reaches H = 139 = 84 + 55   *)
(*                 and H+1 = 140 = 70 + 70 (and beyond, up to 168); the    *)
(*                 worst residue product (-4)(-4) = 16 is reached.         *)
(*  pos_crt_r4     <<16,15,13,11,7>>, K=1, a in -120120..120120 = CRange   *)
(*                 plus H+1, b in {-1,1}: every W tuple of the 5-modulus   *)
(*                 table is reached; SumAbs reaches H and H+1.             *)
(*  pos_kc_r3      <<8,7,5,3>>, AccBits 7, KC = K = 3 = KCMax, a,b in      *)
(*                 -4..4 (every residue of every modulus): a full chunk of *)
(*                 worst products sums to 48 <= 63.                        *)
(*  pos_kc_r2      <<4,3>>, AccBits 5, KC = 3 = KCMax, K = 4 (chunks 3+1,  *)
(*                 carry), a,b in -2..2: worst chunk 12 <= 15 and the      *)
(*                 epilogue add 3 + 12 = 15 = AccMax is exactly tight.     *)
(*  pos_havoc_recon  the neg_chunk parameters (PreChunk dropped, havoc     *)
(*                 reachable: Covers "havoc"), checking only TypeOK,       *)
(*                 CRTRecon and CRTReconAbs: even on garbage planes the    *)
(*                 u8 epilogue keeps W in 0..p-1 and §3.6 returns the      *)
(*                 unique CRT value in CRange.                             *)
(*                                                                         *)
(* NEGATIVE models (must fail as their "\* EXPECT:" line says; each drops  *)
(* exactly one rule, enforced by the spec's Drop ASSUMEs):                 *)
(*  neg_bound      pos_crt_r3_k1 with the guard moved to H+1               *)
(*                 (CRTExactBeyond): a = 420, b = 1 gives z = -420.        *)
(*  neg_chunk      KC = 4 = KCMax+1 (AccBits 7), a,b in {-4,0,4}: four     *)
(*                 products (-4)(-4) = 16 sum to 64 > 63 for p = 8 while   *)
(*                 SumAbs = 64 <= H; havoc (e.g. saturation to 63) breaks  *)
(*                 CRTExact. Note p = 8 = 2^ResBits fails although wrap    *)
(*                 would preserve its residue: .satfinite does not wrap.   *)
(*  neg_resbits    ResBits = 2 with p = 8 = 2 * 2^2: ResiduesFit fails.    *)
(*  neg_resbits_p9 BOUNDARY+1: <<9,8,7,5>>, ResBits = 3, p = 9 = 2^3 + 1:   *)
(*                 SMod(4, 9) = 4 > ResHi = 3, ResiduesFit fails. (Kills   *)
(*                 the off-by-one rule p <= 2^ResBits + 1.)                *)
(*  neg_resbits_p9_store  the same table end to end: the residue 4 is     *)
(*                 stored as the 3-bit int -4 (wrap; saturation would give *)
(*                 3), not congruent mod 9, so CRTExact fails for a = 4,   *)
(*                 b = 1 although every chunk fits (KC = 1 = KCMax). This  *)
(*                 is the real failure mode: operand truncation, not       *)
(*                 accumulator overflow.                                   *)
(*  neg_order      ACCUMULATION ORDER: <<8,7,5,3>>, AccBits 7, KC = K = 5  *)
(*                 (> KCMax = 3), a in {-4,3}, b = -4, and InputFilter =   *)
(*                 "finalSumFits" keeps only inputs whose every chunk sum  *)
(*                 fits (for p = 8: 16j - 12(5-j) <= 63, i.e. not all -4). *)
(*                 E.g. a = <<-4,-4,-4,-4,3>>: sum 52 <= 63 but sum of     *)
(*                 |products| 76 > 63 (sequential .satfinite: 64 -> 63,   *)
(*                 then -12 -> 51). CRTExact fails: the K_c rule must      *)
(*                 bound sum |products|, a "final sum fits" (data-sized)   *)
(*                 rule is unsound. A hardware model whose havoc trigger   *)
(*                 is |final sum| > AccMax would pass this model.          *)
(*  neg_carry      ChunkCarry = FALSE (W = acc mod p of the LAST chunk     *)
(*                 only), K = 2, KC = 1: CRTExact fails.                   *)
(***************************************************************************)
EXTENDS OzakiCRT

ModR2   == <<4, 3>>
ModR3   == <<8, 7, 5, 3>>
ModR3N3 == <<8, 7, 5>>
ModR4   == <<16, 15, 13, 11, 7>>
ModR3P9 == <<9, 8, 7, 5>>          \* negative models only (p = 2^3 + 1)

V420    == -420..420
V120120 == -120120..120120
V12     == -12..12
V7      == -7..7
V4      == -4..4
V2      == -2..2
V1      == -1..1
VPM1    == {-1, 1}
VWorst  == {-4, 0, 4}
VOrdA   == {-4, 3}
VOrdB   == {-4}

CovNone     == {}
CovH        == {"H"}
CovKC       == {"KC"}
CovCarry    == {"carry"}
CovAll      == {"H", "KC", "carry"}
CovHavoc    == {"havoc"}
=============================================================================
