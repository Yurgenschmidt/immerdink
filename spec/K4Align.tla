------------------------------ MODULE K4Align ------------------------------
(***************************************************************************)
(* K4 load alignment (DESIGN v1.13 §4b, stage 2: per-plan K alignment      *)
(* A ∈ {16, 64}, resolved from k alone).                                   *)
(*                                                                         *)
(* What is modelled: every K-tile load address K4 issues for every residue *)
(* row it reads, as an ABSOLUTE byte address, in CUTLASS 2.x's order       *)
(* (predicated_tile_access_iterator.h:204-213, 514-523). For a K-chunk     *)
(* [k0, k0 + kc) of a row that starts at byte `start`:                     *)
(*   the residue tile first, R = ((kc − 1) mod TileK) + 1 elements, at     *)
(*       start + k0,                                                       *)
(*   then the full tiles at start + k0 + R + TileK·i,                      *)
(*       i = 0 .. (kc − R)/TileK − 1.                                      *)
(* `start` is built from the real index arithmetic of every operand K4     *)
(* reads (no "row base" shortcut):                                         *)
(*   RA      region offset (the §4a sum of 256-aligned region sizes,       *)
(*           itself a function of k̂) + ((z0 + z)·m + ī)·k̂, z0 the first   *)
(*           batch of a gridDim.z launch part (z = slot·b + λ);            *)
(*   RB      region offset + (z·n̂ + j0 + j)·k̂, j0 = q·YLen the first      *)
(*           column of a gridDim.y launch part;                            *)
(*   pack A  pack base + res offset + (l·b·r + λ·r + ρ)·k̂ (plane l of a   *)
(*           residue group, any group start);                              *)
(*   pack B  as pack A with ρ = j0 + j < n̂ = ⌈r⌉_AlignQ: the post-swap B   *)
(*           role reads n̂ rows per block at batch stride r·k̂ (rows ≥ r    *)
(*           are the next block's first rows or the zeroed tail).          *)
(* The K extent the GEMM runs is chunked by K_c' (§3.5).                   *)
(*                                                                         *)
(* The rule (stage 1 + stage 2):                                           *)
(*   A(k) = TileK iff k > KPadMin (AUTO: a pure function of the logical k), *)
(*          else AlignQ;                                                   *)
(*   k̂ = ⌈k⌉_A is BOTH the row stride and the GEMM K extent;               *)
(*   K_c' = ⌊K_c / TileK⌋·TileK (131008);                                  *)
(*   region and pack bases are BaseAlign (256)-aligned.                    *)
(* Checked:                                                                *)
(*   Aligned16          every load address is AlignQ-aligned (CUTLASS's   *)
(*                      16-byte rule: correctness, every plan, every       *)
(*                      mutation below keeps it);                          *)
(*   TileStartsAligned  A = TileK ⇒ every residue-tile load is             *)
(*                      TileK-aligned (the chunk start of the row);       *)
(*   MainLoopAligned    A = TileK ⇒ every main-loop load is TileK-aligned  *)
(*                      (after the residue tile: the measured slow path); *)
(*   ExtentOK           the extent covers [0, k), stays in the row        *)
(*                      (extent ≤ stride), is a multiple of AlignQ, and    *)
(*                      its chunks partition it, each ≤ K_c' ≤ K_c;        *)
(*   PadBounded         k̂ − k < A, A ∈ {AlignQ, TileK}.                    *)
(* Mutations (Mut; each negative model must fail, for the stated reason): *)
(*   "strideOnly"     row stride ⌈k⌉_A but GEMM K extent ⌈k⌉_AlignQ: row    *)
(*                    and residue-tile starts stay aligned, the residue    *)
(*                    tile (extent mod TileK) shifts the main loop         *)
(*                    (MainLoopAligned; TileStartsAligned holds)           *)
(*   "partialLanding" strideOnly with the old K_c' = ⌊K_c/AlignQ⌋·AlignQ   *)
(*                    (131056): the draft's "partial landing" trap         *)
(*   "kc32"           K_c' = ⌊K_c / (TileK/2)⌋·(TileK/2) (131040): full    *)
(*                    chunks leave a TileK/2 residue (MainLoopAligned)     *)
(*   "kcOld"          K_c' = ⌊K_c / AlignQ⌋·AlignQ (131056; stage 1 not    *)
(*                    landed) with A = TileK (MainLoopAligned)             *)
(*   "a32"            A = TileK/2 (32) above the threshold: odd rows start *)
(*                    at TileK/2 (TileStartsAligned)                       *)
(*   "base16"         region / pack bases only AlignQ-aligned              *)
(*                    (TileStartsAligned)                                  *)
(* Witnesses (must fail on the design): EveryPlanAligned (A = AlignQ plans, *)
(* k ≤ KPadMin, ARE misaligned: the antecedent matters and the threshold   *)
(* is a real choice), NoMultiChunkA64 (a TileK plan with ≥ 2 chunks is     *)
(* reached, so the K_c' part of the rule is exercised).                    *)
(*                                                                         *)
(* Not modelled: the W / n̂ stride of K4's epilogue and K5 (excluded from   *)
(* §4b), which bytes the loads return (Pipeline: EveryRowByteWritten,      *)
(* NoStaleRead), the perf effect itself (measured, NOTES-K4ALIGN).         *)
(***************************************************************************)
EXTENDS Integers, FiniteSets

CONSTANTS
    TileK,      \* K4 threadblock K tile (real 64 int8 = 64 B)
    AlignQ,     \* CUTLASS operand alignment (real 16 B): the A16 k̂ quantum, n̂'s quantum
    BaseAlign,  \* workspace region / pack base alignment (real 256)
    KC,         \* K_c, the largest exact chunk (real 131071)
    KMax,       \* logical k checked: 1..KMax
    KPadMin,    \* K_PAD64_MIN (real 96)
    Shape,      \* [b, m, n, g, N, MaxZ, YLen] (MC_K4Align.tla)
    Mut         \* "none" or one mutation (header)

Mutations == {"none", "strideOnly", "partialLanding", "kc32", "kcOld", "a32", "base16"}
ASSUME Mut \in Mutations
ASSUME TileK % AlignQ = 0 /\ BaseAlign % TileK = 0 /\ KC >= TileK /\ KPadMin >= 1

Min(x, y)     == IF x < y THEN x ELSE y
CeilDiv(x, d) == (x + d - 1) \div d
RoundUp(x, a) == CeilDiv(x, a) * a

(* ---- the rule ---------------------------------------------------------- *)
\* AUTO: the plan's (and the pack's) K alignment, a pure function of k
APad      == IF Mut = "a32" THEN TileK \div 2 ELSE TileK
AlignOf(k) == IF k > KPadMin THEN APad ELSE AlignQ
Stride(k) == RoundUp(k, AlignOf(k))                       \* k̂: the row stride ...
Extent(k) == IF Mut \in {"strideOnly", "partialLanding"}   \* ... and the GEMM K extent
             THEN RoundUp(k, AlignQ) ELSE Stride(k)
KcP == CASE Mut \in {"kcOld", "partialLanding"} -> (KC \div AlignQ) * AlignQ
         [] Mut = "kc32"                        -> (KC \div (TileK \div 2)) * (TileK \div 2)
         [] OTHER                               -> (KC \div TileK) * TileK
ABase == IF Mut = "base16" THEN AlignQ ELSE BaseAlign    \* region / pack base alignment
Align(x) == RoundUp(x, ABase)

(* ---- chunks and CUTLASS's load sequence -------------------------------- *)
NCh(k)   == CeilDiv(Extent(k), KcP)
K0(c)    == c * KcP
Kc(k, c) == Min(KcP, Extent(k) - K0(c))
Res(kc)  == ((kc - 1) % TileK) + 1                       \* the residue tile, loaded first
TileStart(start, c)    == start + K0(c)
MainLoop(start, k, c)  == {start + K0(c) + Res(Kc(k, c)) + TileK * i :
                              i \in 0..((Kc(k, c) - Res(Kc(k, c))) \div TileK - 1)}

(* ---- the operands' absolute row starts --------------------------------- *)
b == Shape.b
m == Shape.m
n == Shape.n
g == Shape.g
N == Shape.N
NHat == RoundUp(n, AlignQ)
VecBytes(k) == 8 * (2 * m + 2 * n + 2 * k + 3 * b)
\* §4a offsets (in-line operands; the workspace pointer is BaseAlign-aligned:
\* MISALIGNED otherwise; with "base16" a 16-aligned pointer and region rounding)
WsBase  == IF Mut = "base16" THEN AlignQ ELSE 0
OffRA(k) == WsBase + Align(2 * b * m) + Align(2 * b * n) + Align(b * m) + Align(b * n)
                   + Align(VecBytes(k))
OffRB(k) == OffRA(k) + Align(g * b * m * Stride(k))
\* §6b.4 packs (the pack pointer is BaseAlign-aligned: MISALIGNED otherwise)
PackBase == IF Mut = "base16" THEN AlignQ ELSE 0
ResOff(r) == PackBase + Align(2 * b * r) + Align(b * r)
\* Residue groups: the group starting at plane g0 holds GrpSize(g0) moduli;
\* its launches cover z = s·b + λ (slot s), split by gridDim.z into parts of
\* MaxZ batches (the launch's pointer advances by z0 = p·MaxZ batches), and
\* n̂ split by gridDim.y into parts of YLen columns (the B pointer advances by
\* j0 = q·YLen rows).
G0s         == {gi * g : gi \in 0..(CeilDiv(N, g) - 1)}
GrpSize(g0) == Min(g, N - g0)
NZ(g0)      == GrpSize(g0) * b
ZParts(g0)  == 0..(CeilDiv(NZ(g0), Shape.MaxZ) - 1)
ZOf(g0, p)  == {z \in 0..(NZ(g0) - 1) : p * Shape.MaxZ <= z /\ z < (p + 1) * Shape.MaxZ}
YParts      == 0..(CeilDiv(NHat, Shape.YLen) - 1)
YOf(q)      == {j \in 0..(NHat - 1) : q * Shape.YLen <= j /\ j < (q + 1) * Shape.YLen}

\* Each row start = the launch's operand pointer (base + z0 batches [+ j0
\* rows]) + the in-launch batch offset + the row offset, in bytes.
Ops == {"RA", "RB", "PA", "PB"}
RowStarts(op, k) ==
    LET S == Stride(k)  MZ == Shape.MaxZ  YL == Shape.YLen IN
    \* GP: (group start g0, z part p) of every launch; YJ: (y part q, column j)
    LET GP == UNION {{<<g0, p>> : p \in ZParts(g0)} : g0 \in G0s}
        YJ == UNION {{<<q, j>> : j \in YOf(q)} : q \in YParts} IN
    CASE op = "RA" ->     \* RA[slot][λ][ī][k̂]: batch stride m·k̂ (slots reused per group)
           UNION {{OffRA(k) + (gp[2] * MZ) * m * S + ((z - gp[2] * MZ) * m + i) * S :
                     z \in ZOf(gp[1], gp[2]), i \in 0..(m - 1)} : gp \in GP}
      [] op = "RB" ->     \* RB[slot][λ][ĵ][k̂]: batch stride n̂·k̂, y part at row j0
           UNION {{OffRB(k) + (gp[2] * MZ) * NHat * S + (yj[1] * YL) * S
                     + ((z - gp[2] * MZ) * NHat + (yj[2] - yj[1] * YL)) * S :
                       z \in ZOf(gp[1], gp[2]), yj \in YJ} : gp \in GP}
      [] op = "PA" ->     \* pack, post-swap A role (r = m): group g0 at g0·b·r·k̂
           UNION {{ResOff(m) + (gp[1] * b + gp[2] * MZ) * m * S
                     + ((z - gp[2] * MZ) * m + rho) * S :
                       z \in ZOf(gp[1], gp[2]), rho \in 0..(m - 1)} : gp \in GP}
      [] op = "PB" ->     \* pack, post-swap B role (r = n): n̂ rows per block at stride r·k̂
           UNION {{ResOff(n) + (gp[1] * b + gp[2] * MZ) * n * S + (yj[1] * YL) * S
                     + ((z - gp[2] * MZ) * n + (yj[2] - yj[1] * YL)) * S :
                       z \in ZOf(gp[1], gp[2]), yj \in YJ} : gp \in GP}

Chunks(k)  == 0..(NCh(k) - 1)
TileStarts(op, k) == {TileStart(s, c) : s \in RowStarts(op, k), c \in Chunks(k)}
MainLoads(op, k)  == UNION {MainLoop(s, k, c) : s \in RowStarts(op, k), c \in Chunks(k)}
AllLoads(op, k)   == TileStarts(op, k) \cup MainLoads(op, k)

(* ---- the scenario: one (k, operand) per initial state ------------------ *)
VARIABLES kk, op
vars == <<kk, op>>
Init == kk \in 1..KMax /\ op \in Ops
Spec == Init /\ [][UNCHANGED vars]_vars

A64 == AlignOf(kk) = APad /\ kk > KPadMin

Aligned16 == \A a \in AllLoads(op, kk) : a % AlignQ = 0
TileStartsAligned == A64 => \A a \in TileStarts(op, kk) : a % TileK = 0
MainLoopAligned   == A64 => \A a \in MainLoads(op, kk) : a % TileK = 0
ExtentOK ==
    /\ kk <= Extent(kk) /\ Extent(kk) <= Stride(kk) /\ Extent(kk) % AlignQ = 0
    /\ KcP <= KC /\ KcP % AlignQ = 0
    /\ \A c \in Chunks(kk) : Kc(kk, c) \in 1..KcP
    /\ UNION {K0(c)..(K0(c) + Kc(kk, c) - 1) : c \in Chunks(kk)} = 0..(Extent(kk) - 1)
    \* the loads of a chunk cover exactly ceil(kc / TileK) tiles (CUTLASS's iterations)
    /\ \A c \in Chunks(kk) :
         Cardinality(MainLoop(0, kk, c)) + 1 = CeilDiv(Kc(kk, c), TileK)
PadBounded == Stride(kk) - kk < AlignOf(kk) /\ AlignOf(kk) \in {AlignQ, TileK}
\* witnesses (must FAIL on the design)
EveryPlanAligned == \A a \in AllLoads(op, kk) : a % TileK = 0
NoMultiChunkA64  == A64 => NCh(kk) = 1
=============================================================================
