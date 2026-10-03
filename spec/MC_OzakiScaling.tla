------------------------- MODULE MC_OzakiScaling -------------------------
(***************************************************************************)
(* Model values for OzakiScaling. .cfg files cannot hold negative          *)
(* literals or tuples, so they are defined here and bound with `<-`.      *)
(*                                                                         *)
(* Toy formats (FP64 analog: Prec 53, EMin -1022, EMax 1023):              *)
(*   row format    Prec 3, EMin -2, EMax 2: 47 values incl. 3 subnormal   *)
(*                 magnitudes, 5 binades, quantum 2^-4                     *)
(*   wide format   Prec 4, EMin -2, EMax 1: 79 values, quantum 2^-5        *)
(*   tiny format   Prec 2, EMin  0, EMax 1: 11 values (long rows, K = 9)   *)
(*   rnb format    Prec 3, EMin -1, EMax 0, F 4 (exact terms): RN budget *)
(*   kf format     Prec 5, EMin -1, EMax 0, F 8 (exact terms): 63 values  *)
(*   k1 format     Prec 3, EMin -1, EMax 1, F 4 (exact terms): 31 values  *)
(*   joint format  Prec 3, EMin -1, EMax 0: 23 values, quantum 2^-3,      *)
(*                 max 1.75; products span 2^-6 .. 3.06 and K = 2 sums    *)
(*                 reach 6.1, so D hits subnormals, underflow to +-0 and   *)
(*                 overflow to +-Inf. 140185 sorted pair lists.            *)
(*                                                                         *)
(* Toy moduli (P, H = (P-1) div 2, T = floor(sqrt H)):                     *)
(*   <<8>>        P 8,   H 3,   T 1                                         *)
(*   <<4,3>>      P 12,  H 5,   T 2                                         *)
(*   <<4,5>>      P 20,  H 9,   T 3   (T^2 = H: CRT bound exactly tight)   *)
(*   <<8,7>>      P 56,  H 27,  T 5   (greedy ResBits = 3 prefix)          *)
(*   <<8,7,5>>    P 280, H 139, T 11  (greedy ResBits = 3 prefix)          *)
(*   <<4,3,5,7>>  P 420, H 209, T 14  (z up to 8 bits: rich final rounding)*)
(*   <<2,5>>      P 10,  H 4,   T 2   (T^2 = H with T even: RN reaches H) *)
(*   <<8,5>>      P 40,  H 19,  T 4   (K = 4: RN reaches ||a'||^2 = T^2)   *)
(*   <<2,9>>      P 18,  H 8,   T 2   (P/2 = 9 a square: pins H, not P/2) *)
(* All P are even, like the real P (256 is in every table prefix).        *)
(***************************************************************************)
EXTENDS OzakiScaling

MinusOne == -1
MinusTwo == -2

Mod8     == <<8>>
Mod43    == <<4, 3>>
Mod45    == <<4, 5>>
Mod87    == <<8, 7>>
Mod875   == <<8, 7, 5>>
Mod4357  == <<4, 3, 5, 7>>
Mod25    == <<2, 5>>
Mod85    == <<8, 5>>
Mod29    == <<2, 9>>
=============================================================================
