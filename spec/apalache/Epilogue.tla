----------------------------- MODULE Epilogue -----------------------------
(***************************************************************************)
(* DESIGN §3.5: the K-chunked int8 GEMM with the modular epilogue, at the  *)
(* REAL width, symbolically (Apalache), for the real chunk length          *)
(* K_c' = 131008 (DESIGN v1.13: 2047*64; it was 131056) and EVERY real    *)
(* modulus p (p is a symbolic constant ranging over the 20 moduli).        *)
(*                                                                         *)
(* One output element of one plane: chunk after chunk, the int32           *)
(* accumulator acc sums at most K_c' products r*s of int8 residues (any    *)
(* order), then the epilogue reads the u8 plane value W in [0, p) and      *)
(* stores (W + acc) mod p (store-mod for the first chunk is the same with  *)
(* W = 0). The epilogue's input v = W + acc must be exact in int32, and    *)
(* DESIGN states its range: v in [-K_c'*128^2, (p-1) + K_c'*128^2], with   *)
(* max 2146435327 (the true min is -128*127*K_c' = -2129666048; Safety     *)
(* states the looser -K_c'*128^2).                                         *)
(*                                                                         *)
(* Proved as an inductive invariant (spec/apalache/check.sh):              *)
(*   Init => IndInv, IndInv /\ Next => IndInv', IndInv => Safety.          *)
(* Expected violation: IndInv => VMaxTight, i.e. v <= 2146435326 FAILS     *)
(* (counterexample W = 255, acc = 131008 * 128^2, p = 256: reachable, all  *)
(* products (-128)*(-128) after a chunk that left W = 255), so 2146435327 *)
(* is the exact maximum.                                                   *)
(***************************************************************************)
EXTENDS Integers

CONSTANTS
    \* @type: Int;
    Kp,             \* chunk length K_c'
    \* @type: Int;
    p               \* the modulus of this plane

VARIABLES
    \* @type: Int;
    c,              \* products in the current chunk so far
    \* @type: Int;
    acc,            \* exact int32 accumulator of the current chunk
    \* @type: Int;
    W               \* u8 plane value (residue in [0, p))

Int32Min == -2147483648
Int32Max == 2147483647
PMax == 16384       \* (-128)*(-128)
PMin == -16256      \* (-128)*127

\* The 20 greedy moduli (DESIGN §2; spec/params_out.json "moduli").
ConstInit ==
    /\ Kp = 131008
    /\ p \in {256, 255, 253, 251, 247, 241, 239, 233, 229, 227,
              223, 217, 211, 199, 197, 193, 191, 181, 179, 173}

Init == c = 0 /\ acc = 0 /\ W = 0

Next ==
    \/ /\ c < Kp                                   \* one more product
       /\ \E r \in -128..127, s \in -128..127 : acc' = acc + r * s
       /\ c' = c + 1
       /\ W' = W
    \/ /\ W' = (W + acc) % p                       \* epilogue, next chunk
       /\ acc' = 0
       /\ c' = 0

IndInv ==
    /\ c \in 0..Kp
    /\ acc \in (PMin * Kp)..(PMax * Kp)
    /\ PMin * c <= acc
    /\ acc <= PMax * c
    /\ W \in 0..255
    /\ W < p

\* The epilogue may run in any state: its input fits int32 and lies in the
\* DESIGN range.
V == W + acc
Safety ==
    /\ Int32Min <= V /\ V <= Int32Max
    /\ -Kp * 16384 <= V /\ V <= (p - 1) + Kp * 16384
    /\ V <= 2146435327
\* Expected violation: the stated maximum is attained.
VMaxTight == V <= 2146435326
=============================================================================
