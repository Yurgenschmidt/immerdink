---------------------------- MODULE MC_OzakiCRT ----------------------------
(* Model-checking instances for OzakiCRT. Each MC_*.cfg picks one set.   *)
(* pos_*: must pass. neg_*: must FAIL on the named invariant, which shows *)
(* the precondition is necessary and the invariant actually bites.       *)
EXTENDS OzakiCRT

\* Positive: 3 moduli (M = 60), bound 2*3*4 = 24 <= 29, KMax = 15 \div 4 = 3.
Pos3Moduli == <<3, 4, 5>>
\* Positive: 4 moduli (M = 420), bound 2*10*10 = 200 <= 209 (near tight).
Pos4Moduli == <<4, 3, 5, 7>>

\* Negative, bound violated: 2*4*4 = 32 > 29 -> wrap-around in CRT.
\* Negative, overflow: AccBits = 3 (AccMax = 3) but (-2)*(-2) = 4 for m = 4.
\* Negative, residue width: ResBits = 2 holds -2..1, but SMod(_, 5) reaches 2.
NegModuli  == <<3, 4, 5>>
=============================================================================
