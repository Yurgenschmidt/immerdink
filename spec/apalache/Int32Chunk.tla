---------------------------- MODULE Int32Chunk ----------------------------
(***************************************************************************)
(* DESIGN §2 K_c and §3.5: the INT32 accumulator of one K-chunk of the    *)
(* int8 GEMM, at the REAL width, symbolically (Apalache, unbounded SMT     *)
(* integers).                                                              *)
(*                                                                         *)
(* One CUTLASS accumulator adds K products r*s of int8 residues r, s in    *)
(* [-128, 127], in ANY order (any sequence of single-product steps), and   *)
(* with .satfinite every PARTIAL sum must fit int32, not only the final    *)
(* one. The state is (c = products added so far, acc = the exact partial  *)
(* sum). Proved with an inductive invariant (spec/apalache/check.sh):      *)
(*    Init => IndInv,  IndInv /\ Next => IndInv',  IndInv => Safety.       *)
(*  K = 131071 = K_c:  all three hold: every partial sum of <= K_c         *)
(*                     products lies in [-2^31, 2^31 - 1].                 *)
(*  K = 131072 = K_c+1: IndInv => Safety FAILS. The counterexample is      *)
(*                     c = 131072, acc = 131072 * 128^2 = 2^31, the sum of *)
(*                     131072 products (-128)*(-128): plainly reachable    *)
(*                     (every step may pick r = s = -128), so K_c is tight.*)
(* (K_c' = 131056, the chunk length used, is checked with the epilogue in  *)
(* Epilogue.tla.)                                                          *)
(* Exact products are used (r*s, both symbolic), so the bounds -16256*c <= *)
(* acc <= 16384*c are the true extremes, not an over-approximation.        *)
(***************************************************************************)
EXTENDS Integers

CONSTANTS
    \* @type: Int;
    K               \* products per chunk (set by a ConstInit below)

VARIABLES
    \* @type: Int;
    c,              \* products added so far
    \* @type: Int;
    acc             \* exact partial sum

Int32Min == -2147483648
Int32Max == 2147483647
PMax == 16384       \* max r*s = (-128)*(-128) = 128^2
PMin == -16256      \* min r*s = (-128)*127

ConstInitKc      == K = 131071
ConstInitKcPlus1 == K = 131072

Init == c = 0 /\ acc = 0

\* Add one product r*s of two int8 values (any order: every step is one
\* arbitrary product).
Next ==
    /\ c < K
    /\ \E r \in -128..127, s \in -128..127 : acc' = acc + r * s
    /\ c' = c + 1

\* Inductive invariant: the partial sum is within c times the extreme
\* products.
IndInv ==
    /\ c \in 0..K
    /\ acc \in (PMin * K)..(PMax * K)
    /\ PMin * c <= acc
    /\ acc <= PMax * c

\* .satfinite never saturates: every partial sum fits int32.
Safety == Int32Min <= acc /\ acc <= Int32Max
=============================================================================
