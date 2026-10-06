------------------------------- MODULE Pipeline -------------------------------
(***************************************************************************)
(* Execution pipeline and workspace layout of one imd_contract call.       *)
(*                                                                         *)
(* DESIGN.md mapping                                                       *)
(*   §4   kernel sequence on one stream:                                   *)
(*          K0 vec    v1.2: a kernel that generates the scatter vectors    *)
(*                    into the workspace from the fused-mode descriptor    *)
(*                    (its kernel argument); one CTA per tensor's vectors  *)
(*                    here, any interleaving (v1.1: a cudaMemcpyAsync from *)
(*                    the plan's pinned buffer; plan lifetime: HostAPI).   *)
(*                    VecResident (IMD_VEC_RESIDENT): no K0, the vectors   *)
(*                    are in plan-owned device memory ("pvec") and the     *)
(*                    workspace has no vec region                          *)
(*          K1 scale  per A row / B column: σ,τ and non-finite flags;      *)
(*                    K1Split (v1.2 option): two launches, K1a writes γ    *)
(*                    into sA/sB (scratch) and the flags, K1b reads γ back *)
(*                    and overwrites it with σ                             *)
(*          per residue group G (g moduli, ⌈N/g⌉ groups):                  *)
(*            K2 pack A  -> RA[slot][λ][ī][k̂]   (K padded with 0)          *)
(*            K3 pack B  -> RB[slot][λ][ĵ][k̂]   (rows n..n̂ zero)           *)
(*            K4 gemm    per K-chunk c, per gridDim.z launch part and per  *)
(*                       gridDim.y launch part (n̂ split into pieces of     *)
(*                       MaxY N-tiles): epilogue store-mod (c = 0) /       *)
(*                       add-mod (c > 0) into the u8 planes W_l[λ][ī][n̂]   *)
(*          K5 crt    per (λ,ī,j̄): flagged ? native dot : CRT; α/β; C     *)
(*        Fused (DESIGN v1.4 §4, chosen at plan time): ONE launch "KF"     *)
(*                    replaces K0, K1 and group 1's two packs: per A row / *)
(*                    B column and moduli part y it computes σ itself,     *)
(*                    packs the part's slots, and (part 0 only) stores σ   *)
(*                    and the flag; its offsets come from the descriptor   *)
(*                    (kernel argument), never from the vec region, which  *)
(*                    its own vec CTAs write for the later kernels         *)
(*        MParts: gridDim.y of a pack launch (KF and K2/K3): the group's   *)
(*                    slots are split over MParts CTAs per row             *)
(*        FusedLarge (DESIGN v1.6 §4, chosen at plan time per operand):    *)
(*                    for each operand X in FusedLarge ONE launch "KL_X"   *)
(*                    replaces K1 of X and group 1's pack of X: per row of *)
(*                    X one CTA reads the row through the scatter vectors  *)
(*                    (written by K0, a separate launch that has retired), *)
(*                    computes σ itself, packs group 1's slots and stores  *)
(*                    σ and the flag; the other operand keeps K1 and its   *)
(*                    group-1 pack (launched per operand: "scaleop")       *)
(*        Packed (D1, DESIGN §6b, post-swap roles ⊆ {A, B}): operand X is  *)
(*                    read from the caller's packed buffer "pX" (σ, flags, *)
(*                    all N residue planes in the §6b.4 layout, complete   *)
(*                    before the call starts: its RAW / WAR are the        *)
(*                    caller's). No K1, KL, K2/K3 for X; KF scales and     *)
(*                    packs only the in-line operand (both packed: no      *)
(*                    scale or pack kernel at all, K0 still runs: K5's     *)
(*                    native path reads X through the vectors); the        *)
(*                    workspace has no sX, fX, RX (§6b.6). K4 reads plane  *)
(*                    l at l·b·r·k̂, batch λ at stride r·k̂, rows ρ < m      *)
(*                    (role A) or ρ < n̂ (role B: rows r..n̂−1 are the next  *)
(*                    block's first rows or the zeroed tail); K5 reads σ   *)
(*                    and the flags there                                  *)
(*        launch limits (gridDim.z and gridDim.y splits), C must not alias *)
(*        A or B, two                                                      *)
(*        streams without events are a hazard                              *)
(*   §3.7 read rules: β = 0 => C not read; α = 0 or k = 0 => A, B not read *)
(*        (C <- β·C); flagged elements take the native path               *)
(*   §3.5 K-chunks partition k̂, the last one may be shorter; chunk starts  *)
(*        are multiples of 16 (operand alignment)                          *)
(*   §4a  workspace regions, sizes, order, 256-byte aligned offsets,       *)
(*        workspace_size formula (incl. the 0 / vec-only cases)            *)
(*   §4   DIRECT path (native kernel, vec-only workspace), taken in mode   *)
(*        DIRECT and, by descriptor, for every K-empty contraction (k = 1) *)
(*                                                                         *)
(* Data abstraction. Values are replaced by ghost VERSION TAGS that say    *)
(* which write of THIS call a buffer element holds:                        *)
(*   sfA/sfB  -1 = never written in this call (junk); 0/1 = σ final and   *)
(*            the flag K1 computed (σ/τ are written by the same CTA, same  *)
(*            tag); 2/3 = K1a's γ in sA/sB (scratch), flag 0/1 final       *)
(*   vecV     the tensors (A, B, C) whose scatter vectors K0 has written   *)
(*   ra/rb    0 = junk, l = residue of modulus l, Zero = zero padding      *)
(*            (modulus-independent: a zero is a zero for every p_l)        *)
(*   w        set of K POSITIONS h ∈ [0, k̂) whose products are folded     *)
(*            into the u8 plane; {-1} = never written in this call. Kept   *)
(*            for every column j < n̂: for j ≥ n the VALUE is meaningless   *)
(*            (K5 never reads it) but the tag still records which GEMM     *)
(*            writes reached the cell, so the n̂ split is checked to write  *)
(*            every column exactly once per chunk.                         *)
(*   cval     "orig" until K5 writes: "crt" / "native" / "zero" / "betaC"  *)
(*   pw       D1: which rows (and the tail) of each packed buffer imd_pack *)
(*            has written (all, by design, before the call starts). A      *)
(*            packed residue byte decodes, by the §6b.4 layout, to         *)
(*            <<l, λ, ρ>> (modulus l of THIS row: K4 checks the row too,   *)
(*            so a wrong block stride reads a wrong tag), PZero or PJunk.  *)
(* Tagging W with positions (not chunk indices) makes the K-chunking       *)
(* itself checked: a dropped, overlapping or gapped chunk, or store-mod on *)
(* a continuation chunk, leaves a plane ≠ [0, k̂) or folds a position twice.*)
(* The numerical content of each step (residues, exact int32 chunks,       *)
(* store-mod/add-mod = exact sum mod p_l, CRT) is OzakiCRT's (KSplit) and  *)
(* CRTRealization's job; this module checks that every kernel reads the    *)
(* right versions, in the right order, at the right addresses.            *)
(*                                                                         *)
(* What a GEMM really needs from its operands (NoStaleRead checks exactly  *)
(* this, not more):                                                        *)
(*   real positions h < k, real columns j < n: both residues of modulus l; *)
(*   K padding h >= k: at least ONE operand is zero padding (product 0);   *)
(*   B padding rows j >= n: nothing (the W columns j >= n they produce are *)
(*   never read by K5; junk int8 cannot saturate the int32 accumulator     *)
(*   because 128²·K_c' <= 2^31 - 1 counts every position of the chunk).   *)
(* The design zeros more than that (both operands' K padding, B rows       *)
(* n..n̂, every group); pos_pad_minimal shows the minimum suffices and      *)
(* neg_no_pad that the K padding of BOTH operands junk is wrong.          *)
(*                                                                         *)
(* Workspace. The regions are modelled as separate logical buffers. That   *)
(* abstraction is a sound projection of one byte-addressed workspace       *)
(* because of three checked facts:                                         *)
(*   LayoutExact      within each region the address map (logical cell ->  *)
(*                    byte range) is injective and tiles the region with   *)
(*                    no gap: cells·width = bytes and the union of the     *)
(*                    cells' byte ranges is exactly [off, off + bytes);    *)
(*   RegionsDisjoint  regions do not overlap (so cells of different        *)
(*                    regions never share a byte);                         *)
(*   AccessesSafe     every byte a kernel touches lies in its own region,  *)
(*                    in no other region, and below ws_bytes.              *)
(* RegionsUsed adds tightness: in a complete run every non-empty region    *)
(* is touched (no region is allocated for nothing).                        *)
(* Addresses are computed from the §4a offsets with the real element       *)
(* widths (int16 = 2, u8 = 1, int64 = 8) and real index arithmetic; only   *)
(* the alignment constants are scaled (see MC_Pipeline.tla).              *)
(*                                                                         *)
(* GPU semantics. Kernels on one stream run in program order; the CTAs of *)
(* one kernel interleave arbitrarily (each CTA step is atomic: it reads    *)
(* its inputs and writes its outputs); Retire is the kernel boundary, it  *)
(* publishes all writes and records the kernel's completion event. A      *)
(* second stream (packs only) is ordered against the first by events      *)
(* (cudaEventRecord / cudaStreamWaitEvent) via the Deps guard.            *)
(*                                                                         *)
(* Scaling. b, m, n, k, N, g and the chunk/launch limits are tiny; every   *)
(* formula is written with the real index arithmetic, so the same          *)
(* operators are what spec/params.py evaluates at real sizes.              *)
(*                                                                         *)
(* Not modelled (DESIGN gap, see the design_issues of this artifact): the  *)
(* per-batch modulus / Barrett table the batched K4 epilogue reads.       *)
(*                                                                         *)
(* Header note: K0 must precede K1 (K1 gathers A rows via the scatter      *)
(* vectors); the model checks that order, per tensor's vectors.           *)
(* K1 split (DESIGN v1.2 §4): sA/sB first hold γ (an intermediate) and    *)
(* hold σ only after K1b. SigmaFinalAtUse checks that no pack or K5 reads *)
(* γ as σ; NoStaleRead that K1b reads γ (not junk, not σ already written  *)
(* over it).                                                               *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    b, m, n, k,          \* GEMM view (§1.1): batches, rows, cols, contraction length
    N,                   \* number of moduli
    g,                   \* moduli per residue group (1..N)
    ChunkLen,            \* K_c' (real 131056); K-chunk length in k̂ elements
    MaxZ,                \* CUTLASS gridDim.z limit (real 65535)
    TileN,               \* CUTLASS threadblock N (real TB::kN, 128 / 256)
    MaxY,                \* CUTLASS gridDim.y limit in N tiles (real 65535)
    AlignWS,             \* workspace region alignment (real 256 bytes)
    AlignK, AlignN,      \* k̂ = ⌈k⌉_AlignK, n̂ = ⌈n⌉_AlignN (real 16, 16)
    WsBytes,             \* ws_bytes passed to imd_contract
    Direct,              \* mode = IMD_DIRECT
    KEmpty,              \* the descriptor has no K label after lowering
                         \* (outer / Hadamard product; then k = 1)
    VecResident,         \* v1.2 IMD_VEC_RESIDENT: the scatter vectors live in
                         \* plan-owned device memory ("pvec", built at plan
                         \* creation); no K0, no vec region in the workspace
    K1Split,             \* K1 as two launches: γ pass (K1a), then S pass (K1b)
    Fused,               \* v1.4: K0 + K1 + group 1's packs as one launch (KF)
    MParts,              \* v1.4: CTAs per row of a pack launch (moduli parts)
    FusedLarge,          \* v1.6: operands ⊆ {"A", "B"} scaled + packed by KL
    Packed,              \* D1: post-swap roles ⊆ {"A", "B"} read from a packed
                         \* operand (DESIGN §6b)
    TwoStreams,          \* packs on a second stream (NOT v1; future option)
    UseEvents,           \* cross-stream event edges (only with TwoStreams)
    Bug,                 \* "none", one injected design-rule violation, or a
                         \* harmless deviation (see Harmless)
    EnforcePreconditions \* FALSE only in negative models

Bugs == {"none",
         "noWAR",         \* 2 streams: pack(G+1) does not wait for gemm(G) retire
         "storeMod",      \* K-chunk c > 0 uses store-mod instead of add-mod
         "crtEarly",      \* K5 launched before the last group's last GEMM
         "ignoreFB",      \* K5 decides the path from fA only
         "alpha0FullK5",  \* α = 0 skips K1-K4 but K5 runs its normal path
         "beta0ReadsC",   \* K5 reads C even when β = 0
         "aliasCA",       \* C aliases A (row ī of C overlays row ī of A)
         "aliasCB",       \* C aliases B (column j̄ of C overlays column j̄ of B)
         "noPad",         \* packs write no padding at all (K pad of A and B,
                          \* B rows n..n̂)
         "shortRB",       \* RB allocated one alignment unit too small
         "noAlign",       \* region offsets not rounded to AlignWS
         "noSplit",       \* batched GEMM not split at gridDim.z <= MaxZ
         "rbStrideN",     \* RB (slot, λ) block stride uses n rows, not n̂
         "raSizedN",      \* RA sized for N moduli instead of g
         "fullWsK0",      \* k = 0 still allocates every region
         "noYSplit",      \* n̂ not split: one launch with every N tile
         "ySplitOverlap", \* off by one: a y part covers [j0, j0 + YLen]
                          \* (inclusive end), so neighbours share a column
         "ySplitFloor",   \* off by one: floor(n̂ / YLen) y parts, the short
                          \* tail piece is never launched
         "kEmptyEmulate", \* a K-empty contraction is dispatched to the
                          \* emulated path (mode EMULATE) while the
                          \* workspace is sized for DIRECT (align(vec))
         "k0NoC",         \* K0's grid covers only A's and B's vectors
         "residentWsVec", \* resident vectors, but the workspace still has
                          \* a vec region (sized for nothing)
         "residentReadsWs",\* resident vectors, but the kernels read the
                          \* workspace's vec region (not allocated)
         "k1Merged",      \* K1a and K1b CTAs in ONE launch (no kernel
                          \* boundary between the γ pass and the S pass)
         "noSigma",       \* K1b not launched (sA/sB keep γ)
         "sigmaRowSplit", \* K1b splits a row over two CTAs, each writing σ
                          \* over γ in place as soon as it is done
         "packWaitsGamma",\* 2 streams: the packs' K1 event edge is from K1a,
                          \* not K1b
         "fusedReadsVec", \* KF's row CTAs read the scatter vectors from the
                          \* vec region (as K1 / K2 do), which KF's own vec
                          \* CTAs write in the same launch
         "fusedSharedSigma",\* KF's moduli-part CTAs y > 0 read σ from sA
                          \* (written by part 0 of the same launch) instead
                          \* of computing it
         "fusedNoVec",    \* KF has no vec CTAs (K0 dropped, nothing writes
                          \* the vec region)
         "fusedNoSigma",  \* KF uses σ / the flag but does not store them
         "fusedAllGroups",\* KF packs EVERY group into the g slots (one
                          \* launch for all groups)
         "largeWithK0",   \* K0's CTAs run in the same launch as the KL CTAs
                          \* (no kernel boundary between writing the
                          \* vectors and KL reading them)
         "largeRowSplit", \* KL splits a long row over two CTAs (column
                          \* segments); the second takes σ from sA, stored
                          \* by the first in the same launch
         "largeNoSigma",  \* KL uses σ / the flag but does not store them
         "largeAllGroups",\* KL packs EVERY group of its operand into the g
                          \* slots (one launch for all groups)
         "packedWsFull",  \* D1: the workspace still has sX, fX, RX for a
                          \* packed role (sized, never used)
         "packedK1",      \* D1: K1 still scans the packed operand
         "packedPack",    \* D1: K2/K3 still pack the packed operand
         "packedReadsWs", \* D1: K4 reads the packed operand from the
                          \* workspace's RX instead of the packed buffer
         "packedSigmaWs", \* D1: K5 reads σ / the flags of the packed operand
                          \* from the workspace's sX / fX
         "packedBStrideNh",\* D1: role B reads the pack with v1's RB block
                          \* stride n̂·k̂ (not r·k̂)
         "packedNoTail",  \* D1: the packed buffer has no zeroed tail
         "packInFlight",  \* D1: the pack is still being written when the
                          \* call starts (the caller's RAW edge missing)
         "padMinimal"}    \* HARMLESS: K padding only in A and only by the
                          \* first group's pack; B rows n..n̂ never written
\* Deviations from DESIGN that are shown NOT to break anything (positive models).
Harmless == {"padMinimal"}

-----------------------------------------------------------------------------
(* Helpers *)

Min(x, y)      == IF x < y THEN x ELSE y
CeilDiv(x, d)  == (x + d - 1) \div d
RoundUp(x, a)  == CeilDiv(x, a) * a
Range(s)       == {s[i] : i \in DOMAIN s}
RECURSIVE SumSeq(_)
SumSeq(s)      == IF s = <<>> THEN 0 ELSE Head(s) + SumSeq(Tail(s))

-----------------------------------------------------------------------------
(* Problem shape (§1.1, §4a) *)

KHat   == RoundUp(k, AlignK)            \* k̂
NHat   == RoundUp(n, AlignN)            \* n̂
Bs     == 0..(b - 1)
Ms     == 0..(m - 1)
Ns     == 0..(n - 1)
NHs    == 0..(NHat - 1)
KHs    == 0..(KHat - 1)
Ls     == 1..N
Slots  == 0..(g - 1)
RowsA  == Bs \X Ms                      \* (λ, ī)
ColsB  == Bs \X Ns                      \* (λ, j̄)
Elems  == Bs \X Ms \X Ns                \* (λ, ī, j̄) of C
Empty  == b * m * n = 0                 \* BLAS no-op, 0 workspace
\* §4 DIRECT path: "K-empty -> DIRECT is documented dispatch by descriptor".
\* UseDirect is the path the host program takes; SizeDirect is what the
\* §4a formula (imd_workspace_size) assumes. They are the same function of
\* (mode, descriptor) unless a negative model breaks the dispatch.
UseDirect  == Direct \/ (KEmpty /\ Bug # "kEmptyEmulate")
SizeDirect == Direct \/ KEmpty
FullWs == ~Empty /\ ~UseDirect /\ k > 0  \* emulated pipeline with all regions

\* Logical cells of the residue / product buffers.
RADom == Slots \X Bs \X Ms \X KHs       \* RA[slot][λ][ī][h]
RBDom == Slots \X Bs \X NHs \X KHs      \* RB[slot][λ][ĵ][h]
WDom  == Ls \X Bs \X Ms \X NHs          \* W_l[λ][ī][ĵ]

\* Residue groups: group gi holds moduli (gi-1)·g + 1 .. min(gi·g, N).
NG           == CeilDiv(N, g)
GrpSize(gi)  == Min(g, N - (gi - 1) * g)
ModOf(gi, s) == (gi - 1) * g + s + 1    \* modulus held in slot s of group gi

\* K-chunks of k̂ (§3.5): chunk c covers [c·K_c', (c+1)·K_c') ∩ [0, k̂);
\* the last one may be shorter.
NCh       == CeilDiv(KHat, ChunkLen)
Chunk(c)  == {h \in KHs : c * ChunkLen <= h /\ h < (c + 1) * ChunkLen}
AllChunks == 0..(NCh - 1)

\* Batched GEMM launches (§4 launch limits): batch index z = slot·b + λ over
\* (l ∈ G) × λ, split into parts of at most MaxZ.
NZ(gi)        == GrpSize(gi) * b
NParts(gi)    == IF Bug = "noSplit" THEN 1 ELSE CeilDiv(NZ(gi), MaxZ)
PartZ(gi, p)  == IF Bug = "noSplit" THEN 0..(NZ(gi) - 1)
                 ELSE {z \in 0..(NZ(gi) - 1) : p * MaxZ <= z /\ z < (p + 1) * MaxZ}

\* N-tile launches (§4 launch limits, "the N-tile count must be <= 65535
\* (gridDim.y), so n̂ is split across launches if needed"): the GEMM's N
\* extent n̂ is tiled by TileN; one launch has at most MaxY N tiles, i.e.
\* covers YLen = MaxY·TileN columns starting at j0 = q·YLen, the last one
\* shorter, with a partial last tile (CUTLASS predicates the columns past
\* nc). This is src/gemm_k4.cu k4_group (n_per_launch = max_n_tiles·kN,
\* nc = min(n_per_launch, n̂ - j0)); the loop nest is chunk, z part, y part.
YLen          == MaxY * TileN
NYParts       == CASE Bug = "noYSplit"    -> 1
                   [] Bug = "ySplitFloor" -> NHat \div YLen
                   [] OTHER               -> CeilDiv(NHat, YLen)
YStart(q)     == IF Bug = "noYSplit" THEN 0 ELSE q * YLen
YCols(q)      == IF Bug = "noYSplit" THEN NHs
                 ELSE {j \in NHs : YStart(q) <= j
                                  /\ j < YStart(q) + YLen + (IF Bug = "ySplitOverlap" THEN 1 ELSE 0)}
GridY(q)      == CeilDiv(Cardinality(YCols(q)), TileN)     \* gridDim.y of y part q
TileCols(q, y) == {j \in YCols(q) : (j - YStart(q)) \div TileN = y}

-----------------------------------------------------------------------------
(* §4a workspace layout *)

Align(x) == RoundUp(x, AlignWS)

\* int64 scatter vectors, in this order (§4a "vec"):
\*   A: row(m) k(k) batch(b);  B: k(k) col(n) batch(b);  C: row(m) col(n) batch(b)
VA_ROW == 0
VA_K   == m
VA_BAT == m + k
VB_K   == m + k + b
VB_COL == m + 2 * k + b
VB_BAT == m + 2 * k + b + n
VC_ROW == m + 2 * k + 2 * b + n
VC_COL == 2 * m + 2 * k + 2 * b + n
VC_BAT == 2 * m + 2 * k + 2 * b + 2 * n
VecLen == 2 * m + 2 * n + 2 * k + 3 * b         \* = VC_BAT + b
\* The nine sub-vectors as <<start, length>> (lengths from the index sets).
VecSubs == <<<<VA_ROW, m>>, <<VA_K, k>>, <<VA_BAT, b>>,
             <<VB_K, k>>, <<VB_COL, n>>, <<VB_BAT, b>>,
             <<VC_ROW, m>>, <<VC_COL, n>>, <<VC_BAT, b>>>>

Bytes(r) == CASE r = "sA"  -> 2 * b * m
              [] r = "sB"  -> 2 * b * n
              [] r = "fA"  -> 1 * b * m
              [] r = "fB"  -> 1 * b * n
              [] r = "vec" -> 8 * VecLen
              [] r = "RA"  -> (IF Bug = "raSizedN" THEN N ELSE g) * b * m * KHat
              [] r = "RB"  -> g * b * NHat * KHat
              [] r = "W"   -> N * b * m * NHat

Width(r) == CASE r \in {"sA", "sB"} -> 2
              [] r = "vec"           -> 8
              [] OTHER               -> 1

\* Regions present, in layout order (§4a; 0 if b·m·n = 0; vec only for
\* the DIRECT path (mode DIRECT or K-empty) or when k = 0; no vec region
\* with resident vectors). RegionsFor(dir) is the layout of a call that
\* takes (dir = TRUE) or skips the DIRECT path.
VecInWs    == ~VecResident \/ Bug = "residentWsVec"   \* the layout has a vec region
VecReadsWs == ~VecResident \/ Bug = "residentReadsWs" \* kernels read vectors there
\* D1 (§6b.6): the regions sX, fX, RX of a packed post-swap role are absent
\* (0 bytes; same order for the rest).
PackedRegs(X) == IF X = "A" THEN {"sA", "fA", "RA"} ELSE {"sB", "fB", "RB"}
Absent     == IF Bug = "packedWsFull" THEN {} ELSE UNION {PackedRegs(X) : X \in Packed}
AllRegions == SelectSeq(IF VecInWs THEN <<"sA", "sB", "fA", "fB", "vec", "RA", "RB", "W">>
                        ELSE <<"sA", "sB", "fA", "fB", "RA", "RB", "W">>,
                        LAMBDA r : r \notin Absent)
RegionsFor(dir) == IF Empty THEN <<>>
                   ELSE IF dir \/ (k = 0 /\ Bug # "fullWsK0")
                        THEN (IF VecInWs THEN <<"vec">> ELSE <<>>)
                   ELSE AllRegions
\* The layout the kernels of this call use (offsets, accesses).
Regions == RegionsFor(UseDirect)
Active  == Range(Regions)

\* THE FORMULA: workspace_size = Σ_regions align(bytes), for the regions of
\* the path the descriptor dispatches to (SizeDirect).
SizedRegions  == RegionsFor(SizeDirect)
WorkspaceSize == SumSeq([i \in DOMAIN SizedRegions |-> Align(Bytes(SizedRegions[i]))])

\* What the implementation reserves for each region when it computes the
\* offsets (equal to the formula's term unless a negative model injects a bug).
Alloc(r) == CASE Bug = "noAlign"              -> Bytes(r)
              [] Bug = "shortRB" /\ r = "RB"  -> Align(Bytes(r)) - AlignWS
              [] OTHER                        -> Align(Bytes(r))

\* Offsets: regions laid out in order, each at the sum of the previous
\* allocations (cached table: zero-arity).
OffTbl == [r \in Active |->
             LET i == CHOOSE i \in DOMAIN Regions : Regions[i] = r
             IN  SumSeq([j \in 1..(i - 1) |-> Alloc(Regions[j])])]
Off(r) == IF r \in Active THEN OffTbl[r] ELSE 0

\* Plan-owned device bytes (resident vectors): exactly the vec contents.
PlanBytes == IF VecResident /\ ~Empty THEN 8 * VecLen ELSE 0

\* D1 packed buffer of post-swap role X (DESIGN §6b.4, format 1), caller-
\* owned, its own address space (offset 0 = the `packed` pointer):
\*   sig   2·b·r                     int16 σ of row (λ, ρ) at λ·r + ρ
\*   flag  1·b·r                     u8 flag, same index
\*   res   N·b·r·k̂ + (⌈r⌉ − r)·k̂     plane l (1..N) of row (λ, ρ), position h
\*                                   at (l−1)·b·r·k̂ + (λ·r + ρ)·k̂ + h; h ≥ k:
\*                                   0; then the zeroed tail
\*   pack_bytes = Σ align(region), 0 if b·r·k = 0
\* r is X's row count (m for role A, n for role B), ⌈r⌉ = ⌈r⌉_AlignN (16:
\* the GEMM's n̂ rounding), the alignment is the workspace's (256).
PR(X)       == IF X = "A" THEN m ELSE n
PRows(X)    == Bs \X (0..(PR(X) - 1))
PTailRow    == <<-1, -1>>                       \* pw index of the tail
PTail(X)    == IF Bug = "packedNoTail" THEN 0 ELSE (RoundUp(PR(X), AlignN) - PR(X)) * KHat
PSubs       == <<"sig", "flag", "res">>
PSubBytes(X, s) == CASE s = "sig"  -> 2 * b * PR(X)
                     [] s = "flag" -> b * PR(X)
                     [] s = "res"  -> N * b * PR(X) * KHat + PTail(X)
PSubOff(X, s)   == CASE s = "sig"  -> 0
                     [] s = "flag" -> Align(PSubBytes(X, "sig"))
                     [] s = "res"  -> Align(PSubBytes(X, "sig")) + Align(PSubBytes(X, "flag"))
PackBytes(X) == IF b * PR(X) * k = 0 THEN 0
                ELSE SumSeq([i \in 1..3 |-> Align(PSubBytes(X, PSubs[i]))])
\* Access region names of the packed buffers' sub-regions.
PSub(X, s) == CASE X = "A" /\ s = "sig"  -> "pAsig"  [] X = "A" /\ s = "flag" -> "pAflag"
                [] X = "A" /\ s = "res"  -> "pAres"  [] X = "B" /\ s = "sig"  -> "pBsig"
                [] X = "B" /\ s = "flag" -> "pBflag" [] X = "B" /\ s = "res"  -> "pBres"
PackRegs   == {PSub(X, s) : X \in Packed, s \in {"sig", "flag", "res"}}
PRegOf     == [r \in PackRegs |-> CHOOSE xs \in Packed \X {"sig", "flag", "res"} : PSub(xs[1], xs[2]) = r]
\* §6b.4 addresses (the layout contract)
PResRel(X, l, lam, rho, h) == (l - 1) * b * PR(X) * KHat + (lam * PR(X) + rho) * KHat + h
AccPSig(X, lam, rho)  == <<PSub(X, "sig"), PSubOff(X, "sig") + 2 * (lam * PR(X) + rho), 2>>
AccPFlag(X, lam, rho) == <<PSub(X, "flag"), PSubOff(X, "flag") + lam * PR(X) + rho, 1>>
AccPRes(X, rel)       == <<PSub(X, "res"), PSubOff(X, "res") + rel, 1>>
\* K4's read of the packed operand X (§6b.4 "Role A / Role B"): plane l,
\* batch λ at stride r·k̂, row ρ (< m in role A, < n̂ in role B), position h.
\* "packedBStrideNh": role B read as v1's RB, block (l, λ) at n̂ rows.
PK4Rel(X, l, lam, rho, h) ==
    IF X = "B" /\ Bug = "packedBStrideNh"
    THEN (((l - 1) * b + lam) * NHat + rho) * KHat + h
    ELSE (l - 1) * b * PR(X) * KHat + (lam * PR(X) + rho) * KHat + h

\* Byte addresses of the elements kernels touch (access = <<region, addr, width>>).
\* Resident vectors are addressed in the plan's buffer "pvec" (offset 0).
RBRows == IF Bug = "rbStrideN" THEN n ELSE NHat   \* rows per (slot, λ) block of RB
AccVec(i)          == IF VecReadsWs THEN <<"vec", Off("vec") + 8 * i, 8>>
                      ELSE <<"pvec", 8 * i, 8>>
AccSA(lam, i)      == <<"sA", Off("sA") + 2 * (lam * m + i), 2>>
AccSB(lam, j)      == <<"sB", Off("sB") + 2 * (lam * n + j), 2>>
AccFA(lam, i)      == <<"fA", Off("fA") + lam * m + i, 1>>
AccFB(lam, j)      == <<"fB", Off("fB") + lam * n + j, 1>>
AddrRA(s, lam, i, h) == Off("RA") + ((s * b + lam) * m + i) * KHat + h
AddrRB(s, lam, j, h) == Off("RB") + ((s * b + lam) * RBRows + j) * KHat + h
AddrW(l, lam, i, j)  == Off("W") + (((l - 1) * b + lam) * m + i) * NHat + j
AccRA(s, lam, i, h)  == <<"RA", AddrRA(s, lam, i, h), 1>>
AccRB(s, lam, j, h)  == <<"RB", AddrRB(s, lam, j, h), 1>>
AccW(l, lam, i, j)   == <<"W", AddrW(l, lam, i, j), 1>>

\* Gather/scatter vector entries read for an A row, a B column, a C element.
VecA(lam, i)    == {AccVec(VA_ROW + i), AccVec(VA_BAT + lam)}
                   \cup {AccVec(VA_K + h) : h \in 0..(k - 1)}
VecB(lam, j)    == {AccVec(VB_COL + j), AccVec(VB_BAT + lam)}
                   \cup {AccVec(VB_K + h) : h \in 0..(k - 1)}
VecC(lam, i, j) == {AccVec(VC_ROW + i), AccVec(VC_COL + j), AccVec(VC_BAT + lam)}

\* Every logical cell of a region (as an access), and how many there are,
\* counted from the logical index sets (independently of the addresses).
Cells(r) ==
    CASE r = "sA"  -> {AccSA(x[1], x[2]) : x \in RowsA}
      [] r = "sB"  -> {AccSB(x[1], x[2]) : x \in ColsB}
      [] r = "fA"  -> {AccFA(x[1], x[2]) : x \in RowsA}
      [] r = "fB"  -> {AccFB(x[1], x[2]) : x \in ColsB}
      [] r = "vec" -> UNION {{AccVec(VecSubs[v][1] + x) : x \in 0..(VecSubs[v][2] - 1)}
                                : v \in 1..9}
      [] r = "RA"  -> {AccRA(x[1], x[2], x[3], x[4]) : x \in RADom}
      [] r = "RB"  -> {AccRB(x[1], x[2], x[3], x[4]) : x \in RBDom}
      [] r = "W"   -> {AccW(x[1], x[2], x[3], x[4]) : x \in WDom}
NCells(r) ==
    CASE r \in {"sA", "fA"} -> Cardinality(RowsA)
      [] r \in {"sB", "fB"} -> Cardinality(ColsB)
      [] r = "vec"          -> SumSeq([v \in 1..9 |-> VecSubs[v][2]])
      [] r = "RA"           -> Cardinality(RADom)
      [] r = "RB"           -> Cardinality(RBDom)
      [] r = "W"            -> Cardinality(WDom)
ByteSet(r) == UNION {x[2]..(x[2] + x[3] - 1) : x \in Cells(r)}

\* An access is safe iff it lies inside its own region, overlaps no other
\* region, and ends at or below ws_bytes.
AccOK(x) ==
    LET r == x[1]  a == x[2]  wd == x[3] IN
    IF r = "pvec" THEN 0 <= a /\ a + wd <= PlanBytes   \* inside the plan's buffer
    ELSE IF r \in PackRegs                              \* inside its packed sub-region
    THEN LET X == PRegOf[r][1]  s == PRegOf[r][2] IN
         /\ PSubOff(X, s) <= a /\ a + wd <= PSubOff(X, s) + PSubBytes(X, s)
         /\ a + wd <= PackBytes(X)
    ELSE
    /\ r \in Active
    /\ Off(r) <= a /\ a + wd <= Off(r) + Bytes(r)
    /\ a + wd <= WsBytes
    /\ \A r2 \in Active \ {r} :
          Bytes(r2) = 0 \/ a + wd <= Off(r2) \/ Off(r2) + Bytes(r2) <= a
OOB(S)  == IF \A x \in S : AccOK(x) THEN {} ELSE {"oob"}
Regs(S) == {x[1] : x \in S}

-----------------------------------------------------------------------------
(* Host program (§4): a constant kernel sequence; which one runs depends   *)
(* only on host-side facts (mode, k, and whether α = 0).                   *)

KrQ(kd, gi, c, p, q) == [kind |-> kd, gi |-> gi, c |-> c, p |-> p, q |-> q]
Kr(kd, gi, c, p)     == KrQ(kd, gi, c, p, 0)
KVec    == Kr("vec", 0, 0, 0)
KScale  == Kr("scale", 0, 0, 0)
KGamma  == Kr("gamma", 0, 0, 0)       \* K1a
KSigma  == Kr("sigma", 0, 0, 0)       \* K1b
\* D1: the operands this call scales and packs itself (in-line). A packed
\* operand gets no K1, KL or K2/K3 ("packedK1" / "packedPack": it still does).
InA == "A" \notin Packed
InB == "B" \notin Packed
\* K1 as launched: one kernel (v1.1), or K1a then K1b (K1Split); none when
\* both operands are packed.
K1Seq == CASE InA = FALSE /\ InB = FALSE /\ Bug # "packedK1" -> <<>>
           [] ~K1Split          -> <<KScale>>
           [] Bug = "k1Merged"  -> <<Kr("k1merged", 0, 0, 0)>>
           [] Bug = "noSigma"   -> <<KGamma>>
           [] OTHER             -> <<KGamma, KSigma>>
\* The kernel whose retirement makes σ/τ final (the packs' RAW edge, 2 streams).
K1FinalKind == CASE ~K1Split               -> "scale"
                 [] Bug = "packWaitsGamma" -> "gamma"
                 [] OTHER                  -> "sigma"
KCrt    == Kr("crt", 0, 0, 0)
KDirect == Kr("direct", 0, 0, 0)
KFused  == Kr("fused", 1, 0, 0)       \* KF: K0 + K1 + packs of group 1 (v1.4)

\* For chunk c, for z part p, for y part q: one batched GEMM launch.
GemmSeq(gi) == LET P == NParts(gi)  Q == NYParts IN
               [i \in 1..(NCh * P * Q) |->
                  KrQ("gemm", gi, (i - 1) \div (P * Q), ((i - 1) \div Q) % P, (i - 1) % Q)]
PackOpA(gi) == IF InA \/ Bug = "packedPack" THEN <<Kr("packA", gi, 0, 0)>> ELSE <<>>
PackOpB(gi) == IF InB \/ Bug = "packedPack" THEN <<Kr("packB", gi, 0, 0)>> ELSE <<>>
RECURSIVE GroupsFrom(_)
GroupsFrom(gi) == IF gi > NG THEN <<>>
                  ELSE PackOpA(gi) \o PackOpB(gi)
                       \o GemmSeq(gi) \o GroupsFrom(gi + 1)
GP == GroupsFrom(1)
\* Every group's GEMMs and nothing else (the "fusedAllGroups" negative: one
\* KF launch packed every group into the same g slots beforehand).
RECURSIVE GemmSeqsFrom(_)
GemmSeqsFrom(gi) == IF gi > NG THEN <<>> ELSE GemmSeq(gi) \o GemmSeqsFrom(gi + 1)

\* K0 runs unless the vectors are resident in the plan.
K0Seq == IF VecResident THEN <<>> ELSE <<KVec>>
\* v1.4 fused path: KF, then group 1's GEMMs, then groups 2.. as usual
\* (their packs read the σ / τ and vectors KF stored).
\* D1: KF scales and packs only the in-line operand; with both operands
\* packed there is no KF (no scale or pack kernel at all): K0, the GEMMs.
FusedGP == IF Bug = "fusedAllGroups" THEN <<KFused>> \o GemmSeqsFrom(1)
           ELSE IF ~InA /\ ~InB THEN K0Seq \o GemmSeqsFrom(1)
           ELSE <<KFused>> \o GemmSeq(1) \o GroupsFrom(2)
\* v1.6 fused large path: K0, then per operand (A, then B) one launch: KL_X
\* (K1 of X + group 1's pack of X) if X ∈ FusedLarge, else K1 of X alone
\* ("scaleop"); then group 1's packs of the other operands, group 1's GEMMs,
\* and groups 2.. as usual (their packs read the σ / τ that KL stored).
\* Field p of a "large" / "scaleop" kernel is the operand (0 = A, 1 = B).
OpIdx(X)    == IF X = "A" THEN 0 ELSE 1
KLarge(X)   == Kr("large", 1, 0, OpIdx(X))
KScaleOp(X) == Kr("scaleop", 0, 0, OpIdx(X))
OpKernel(X) == IF X \in FusedLarge THEN KLarge(X) ELSE KScaleOp(X)
\* "largeWithK0": K0's CTAs and every KL CTA in ONE launch ("largevec").
LargeHead == IF Bug = "largeWithK0"
             THEN <<Kr("largevec", 1, 0, 0)>>
                  \o (IF "A" \in FusedLarge THEN <<>> ELSE <<KScaleOp("A")>>)
                  \o (IF "B" \in FusedLarge THEN <<>> ELSE <<KScaleOp("B")>>)
             ELSE K0Seq \o (IF InA THEN <<OpKernel("A")>> ELSE <<>>)
                        \o (IF InB THEN <<OpKernel("B")>> ELSE <<>>)
\* No pack of a KL operand in group 1 (in the "largeAllGroups" negative: in
\* no group, KL having packed them all), none of a packed operand.
LargeSkips(X, gi) == X \in FusedLarge /\ (gi = 1 \/ Bug = "largeAllGroups")
PacksL(gi) == (IF LargeSkips("A", gi) \/ ~InA THEN <<>> ELSE <<Kr("packA", gi, 0, 0)>>)
              \o (IF LargeSkips("B", gi) \/ ~InB THEN <<>> ELSE <<Kr("packB", gi, 0, 0)>>)
RECURSIVE GroupsL(_)
GroupsL(gi) == IF gi > NG THEN <<>> ELSE PacksL(gi) \o GemmSeq(gi) \o GroupsL(gi + 1)
ProgFull == IF Empty THEN <<>>
            ELSE IF UseDirect THEN K0Seq \o <<KDirect>>
            ELSE IF Fused THEN FusedGP \o <<KCrt>>
            ELSE IF FusedLarge # {} THEN LargeHead \o GroupsL(1) \o <<KCrt>>
            ELSE IF Bug = "crtEarly"
                 THEN K0Seq \o K1Seq \o SubSeq(GP, 1, Len(GP) - 1)
                      \o <<KCrt>> \o SubSeq(GP, Len(GP), Len(GP))
            ELSE K0Seq \o K1Seq \o GP \o <<KCrt>>
\* α = 0 or k = 0: C <- β·C only (§3.7); K1-K4 are not launched.
ProgTriv == IF Empty THEN <<>> ELSE K0Seq \o <<IF UseDirect THEN KDirect ELSE KCrt>>

\* Indexed by "α = 0" (a host-side fact). Cached tables (zero-arity).
Prog == [a \in BOOLEAN |-> IF a \/ k = 0 THEN ProgTriv ELSE ProgFull]

Streams      == IF TwoStreams THEN {"main", "side"} ELSE {"main"}
StreamOf(kr) == IF TwoStreams /\ kr.kind \in {"packA", "packB"} THEN "side" ELSE "main"
\* SProg[a][st]: global indices of the kernels enqueued on stream st, in order.
SProg == [a \in BOOLEAN |-> [st \in Streams |->
            SelectSeq([i \in DOMAIN Prog[a] |-> i],
                      LAMBDA i : StreamOf(Prog[a][i]) = st)]]

\* Cross-stream event edges (two streams only): a pack waits for K1 (RAW on
\* σ, vec; with K1Split from K1b, the launch that makes σ final) and for every GEMM of the previous group (WAR on RA/RB reuse); a
\* GEMM waits for both packs of its group (RAW on RA/RB).
Deps == [a \in BOOLEAN |-> [i \in DOMAIN Prog[a] |->
           LET P == Prog[a]  Ki == P[i] IN
           IF ~(TwoStreams /\ UseEvents) THEN {}
           ELSE CASE Ki.kind \in {"packA", "packB"} ->
                       {j \in DOMAIN P : P[j].kind = K1FinalKind}
                       \cup (IF Ki.gi > 1 /\ Bug # "noWAR"
                             THEN {j \in DOMAIN P : P[j].kind = "gemm" /\ P[j].gi = Ki.gi - 1}
                             ELSE {})
                  [] Ki.kind = "gemm" ->
                       {j \in DOMAIN P : P[j].kind \in {"packA", "packB"} /\ P[j].gi = Ki.gi}
                  [] OTHER -> {}]]

\* CTAs of each kernel (any grid; what matters is which outputs each owns).
K0CTAs == IF Bug = "k0NoC" THEN {"A", "B"} ELSE {"A", "B", "C"}
K1CTAs == (IF InA \/ Bug = "packedK1" THEN {<<"a", x[1], x[2]>> : x \in RowsA} ELSE {})
          \cup (IF InB \/ Bug = "packedK1" THEN {<<"b", x[1], x[2]>> : x \in ColsB} ELSE {})
\* v1.4 moduli parts: CTA part y of a pack launch writes the slots
\* [y·|G| div MParts, (y+1)·|G| div MParts) (contract.cu: blockIdx.y of gridDim.y).
Parts == 0..(MParts - 1)
PartSlots(gi, y) == {s \in 0..(GrpSize(gi) - 1) :
                       (y * GrpSize(gi)) \div MParts <= s /\ s < ((y + 1) * GrpSize(gi)) \div MParts}
\* KF's CTAs: per A row / B column (padded columns included) and moduli part,
\* plus one per tensor's vectors (none with resident vectors).
FusedVecCTAs == IF VecResident \/ Bug = "fusedNoVec" THEN {} ELSE {<<"v", X>> : X \in {"A", "B", "C"}}
FusedCTAs == (IF InA THEN {<<"a", x[1], x[2], y>> : x \in RowsA, y \in Parts} ELSE {})
             \cup (IF InB THEN {<<"b", lam, j, y>> : lam \in Bs, j \in NHs, y \in Parts} ELSE {})
             \cup FusedVecCTAs
\* v1.6 KL of operand p: one CTA per A row / (padded) B column and row
\* segment; one segment (a CTA owns its whole row) except in the
\* "largeRowSplit" negative.
Segs == IF Bug = "largeRowSplit" THEN {0, 1} ELSE {0}
LargeCTAs(p) == IF p = 0 THEN {<<"a", x[1], x[2], s>> : x \in RowsA, s \in Segs}
                ELSE {<<"b", lam, j, s>> : lam \in Bs, j \in NHs, s \in Segs}
\* K1 of one operand (the operand not in FusedLarge).
ScaleOpCTAs(p) == IF p = 0 THEN {<<"a", x[1], x[2]>> : x \in RowsA}
                  ELSE {<<"b", x[1], x[2]>> : x \in ColsB}
CTAsOf(kr) ==
    CASE kr.kind = "vec"    -> K0CTAs                 \* one per tensor's vectors
      [] kr.kind \in {"scale", "gamma"} -> K1CTAs     \* one per A row / B column
      [] kr.kind = "sigma"  -> IF Bug = "sigmaRowSplit"
                               THEN {<<t[1], t[2], t[3], hf>> : t \in K1CTAs, hf \in {0, 1}}
                               ELSE K1CTAs
      [] kr.kind = "k1merged" -> {<<"g">> \o t : t \in K1CTAs} \cup {<<"s">> \o t : t \in K1CTAs}
      [] kr.kind = "packA"  -> {<<x[1], x[2], y>> : x \in RowsA, y \in Parts}  \* A row, moduli part
      [] kr.kind = "packB"  -> Bs \X NHs \X Parts       \* one (padded) B column and part each
      [] kr.kind = "fused"  -> FusedCTAs
      [] kr.kind = "scaleop" -> ScaleOpCTAs(kr.p)
      [] kr.kind = "large"  -> LargeCTAs(kr.p)
      [] kr.kind = "largevec" -> {<<"v", X>> : X \in K0CTAs}
                                 \cup UNION {LargeCTAs(OpIdx(X)) : X \in FusedLarge}
      [] kr.kind = "gemm"   -> PartZ(kr.gi, kr.p) \X Ms \X (0..(GridY(kr.q) - 1))
                                                        \* (batch z, row ī, N tile y)
      [] kr.kind \in {"crt", "direct"} -> Elems         \* one C element each
CTATbl == [a \in BOOLEAN |-> [i \in DOMAIN Prog[a] |-> CTAsOf(Prog[a][i])]]

-----------------------------------------------------------------------------
(* Assumptions *)

WellFormed ==
    /\ \A x \in {b, m, n, k} : x \in Nat
    /\ N \in Nat /\ N >= 1 /\ g \in 1..N
    /\ ChunkLen >= 1 /\ MaxZ >= 1 /\ TileN >= 1 /\ MaxY >= 1
    /\ AlignWS >= 1 /\ AlignK >= 1 /\ AlignN >= 1
    /\ WsBytes \in Nat
    /\ {Direct, KEmpty, K1Split, TwoStreams, UseEvents, EnforcePreconditions} \subseteq BOOLEAN
    /\ KEmpty => k = 1                     \* the K-empty GEMM view has k = 1
    /\ Bug \in Bugs
    /\ Bug \in {"k1Merged", "noSigma", "sigmaRowSplit", "packWaitsGamma"} => K1Split
    /\ Bug = "packWaitsGamma" => TwoStreams /\ UseEvents
    /\ VecResident \in BOOLEAN
    /\ Bug \in {"residentWsVec", "residentReadsWs"} => VecResident
    /\ Fused \in BOOLEAN /\ MParts \in Nat /\ MParts >= 1
    /\ Fused => ~TwoStreams /\ ~K1Split          \* v1.4: single stream, one-launch K1
    /\ Bug \in {"fusedReadsVec", "fusedSharedSigma", "fusedNoVec", "fusedNoSigma",
                "fusedAllGroups"} => Fused
    /\ Bug = "fusedNoVec" => ~VecResident
    /\ FusedLarge \subseteq {"A", "B"}
    /\ FusedLarge # {} => ~Fused /\ ~K1Split /\ ~TwoStreams   \* v1.6: single stream
    /\ Bug \in {"largeWithK0", "largeRowSplit", "largeNoSigma", "largeAllGroups"}
          => FusedLarge # {}
    /\ Bug = "largeWithK0" => ~VecResident
    \* D1 packed slots: single stream, one-launch K1 (v1); a packed operand is
    \* never KL's; a K-empty pack does not exist (§6b.3: NOT_SUPPORTED).
    /\ Packed \subseteq {"A", "B"}
    /\ Packed # {} => ~K1Split /\ ~TwoStreams /\ ~KEmpty
    /\ FusedLarge \cap Packed = {}
    /\ Bug \in {"packedWsFull", "packedK1", "packedPack", "packedReadsWs", "packedSigmaWs",
                "packedBStrideNh", "packedNoTail", "packInFlight"} => Packed # {}
    /\ Bug \in {"packedBStrideNh", "packedNoTail"} => "B" \in Packed
    /\ Bug \in {"fusedReadsVec", "fusedSharedSigma", "fusedNoVec", "fusedNoSigma",
                "fusedAllGroups", "largeWithK0", "largeRowSplit", "largeNoSigma",
                "largeAllGroups"} => Packed = {}

\* Preconditions of the design; each has a boundary-tight positive model and
\* a negative model at boundary + 1 (see MC_Pipeline.tla).
Preconditions ==
    /\ WsBytes >= WorkspaceSize                \* caller passes enough workspace
    /\ ChunkLen % AlignK = 0                   \* §3.5: K_c' multiple of 16
    /\ AlignWS % 8 = 0                         \* regions naturally aligned (int64)
    /\ AlignWS % AlignK = 0 /\ AlignWS % AlignN = 0  \* 256 % 16 = 0
    /\ TileN % AlignN = 0                      \* N tile: a multiple of 16
\* Design rules (removing one is what the Bug-models do; Harmless deviations
\* are allowed in positive models).
DesignRules == (Bug = "none" \/ Bug \in Harmless) /\ (TwoStreams => UseEvents)

ASSUME WellFormed
ASSUME EnforcePreconditions => Preconditions /\ DesignRules

-----------------------------------------------------------------------------
VARIABLES
    pc,        \* pc[st]: position in SProg of the kernel running on stream st
    done,      \* done[st]: finished CTAs of that kernel
    retired,   \* indices of retired kernels (= recorded completion events)
    alpha0,    \* α = 0 (input)
    beta0,     \* β = 0 (input)
    flagA,     \* flagA[λ,ī]: row ī of A_λ has Inf/NaN (input data)
    flagB,     \* flagB[λ,j̄]: column j̄ of B_λ has Inf/NaN (input data)
    aDirty,    \* aDirty[λ,ī]: row of A overwritten during the call (aliasing)
    bDirty,    \* bDirty[λ,j̄]: column of B overwritten during the call (aliasing)
    vecV,      \* tensors whose scatter vectors K0 wrote in this call
    sfA, sfB,  \* σ/fA and τ/fB tags (-1 junk, else the flag written by K1)
    ra, rb,    \* residue buffer tags (0 junk, l = modulus l, Zero = padding)
    w,         \* W plane tags (set of folded K positions, {-1} junk)
    cw,        \* cw[e]: number of writes to C[e]
    cval,      \* cval[e]: what C[e] holds
    touched,   \* ghost: workspace regions some kernel accessed
    err,       \* ghost: kinds of violations observed
    pw         \* D1: pw[X][λ, ρ]: row (λ, ρ) of packed operand X written by
               \* imd_pack (σ, flag, all planes); pw[X][PTailRow]: the tail

vars    == <<pc, done, retired, alpha0, beta0, flagA, flagB, aDirty, bDirty,
             vecV, sfA, sfB, ra, rb, w, cw, cval, touched, err, pw>>
\* What the call's kernels never write: its inputs (the packed buffers too).
inputs  == <<alpha0, beta0, flagA, flagB, pw>>

Zero     == -1                            \* ra/rb tag of zero padding
ErrKinds == {"stale", "gamma", "plane", "dup", "reuse", "oob", "readC", "readAB", "clobber"}
CTags    == {"orig", "zero", "betaC", "native", "crt"}

Init ==
    /\ alpha0 \in BOOLEAN /\ beta0 \in BOOLEAN
    /\ flagA \in [RowsA -> BOOLEAN] /\ flagB \in [ColsB -> BOOLEAN]
    /\ pc = [st \in Streams |-> 1] /\ done = [st \in Streams |-> {}]
    /\ retired = {}
    /\ aDirty = [x \in RowsA |-> FALSE] /\ bDirty = [x \in ColsB |-> FALSE]
    /\ vecV = IF VecResident THEN {"A", "B", "C"} ELSE {}   \* resident: built at plan creation
    /\ sfA = [x \in RowsA |-> -1] /\ sfB = [x \in ColsB |-> -1]
    /\ ra = [x \in RADom |-> 0] /\ rb = [x \in RBDom |-> 0]
    /\ w = [x \in WDom |-> {-1}]
    /\ cw = [e \in Elems |-> 0] /\ cval = [e \in Elems |-> "orig"]
    /\ touched = {}
    /\ err = {}
    \* The packed buffers are complete before the call starts (the caller's
    \* RAW edge); "packInFlight": imd_pack is still writing them.
    /\ pw = [X \in Packed |-> [x \in PRows(X) \cup {PTailRow} |-> Bug # "packInFlight"]]

\* "packInFlight": imd_pack (another stream, no event) writes one row (or the
\* tail) of a packed buffer at any time, interleaved with the call.
PackWrite ==
    /\ \E X \in Packed : \E x \in PRows(X) \cup {PTailRow} :
          /\ ~pw[X][x]
          /\ pw' = [pw EXCEPT ![X][x] = TRUE]
    /\ UNCHANGED <<pc, done, retired, alpha0, beta0, flagA, flagB, aDirty, bDirty,
                   vecV, sfA, sfB, ra, rb, w, cw, cval, touched, err>>

-----------------------------------------------------------------------------
(* Kernel bodies: one atomic CTA step each *)

\* A, B may not be read when α = 0 or k = 0 (§3.7).
ABForbidden == alpha0 \/ k = 0
ReadAB      == IF ABForbidden THEN {"readAB"} ELSE {}
\* K5/DIRECT in "C <- β·C" mode.
TrivialK5   == ABForbidden /\ ~(Bug = "alpha0FullK5" /\ alpha0)
\* K5 reads C iff β ≠ 0 (§3.7); reading it after it was written is stale.
ReadsC      == ~beta0 \/ Bug = "beta0ReadsC"
CReadErr(e) == (IF beta0 /\ ReadsC THEN {"readC"} ELSE {})
               \cup (IF ReadsC /\ cw[e] > 0 THEN {"stale"} ELSE {})
\* Reading an A row / B column that a C write has overwritten (aliasing).
ClobA(lam, i) == IF aDirty[<<lam, i>>] THEN {"clobber"} ELSE {}
ClobB(lam, j) == IF bDirty[<<lam, j>>] THEN {"clobber"} ELSE {}
\* WAR on reused RA/RB: some GEMM of group gi has not retired.
PendingGemm(gi) == \E j \in DOMAIN Prog[alpha0] :
                      Prog[alpha0][j].kind = "gemm" /\ Prog[alpha0][j].gi = gi
                      /\ j \notin retired
Reuse(gi) == IF gi > 1 /\ PendingGemm(gi - 1) THEN {"reuse"} ELSE {}
Stale(c)  == IF c THEN {"stale"} ELSE {}
\* σ/τ as read by a pack or K5: junk is stale; K1a's γ read as σ is "gamma".
SigmaRead(v) == (IF v = -1 THEN {"stale"} ELSE {}) \cup (IF v \in {2, 3} THEN {"gamma"} ELSE {})
FlagOf(v)    == v \in {1, 3}          \* the flag is final from K1a (or K1) on
\* Record the workspace accesses of a step: bounds and the touched regions.
Access(S, es) == /\ err' = err \cup es \cup OOB(S)
                /\ touched' = touched \cup Regs(S)

\* K0, CTA X ∈ {A, B, C}: generate tensor X's three scatter vectors (from the
\* descriptor in the kernel argument; no input is read).
VecSubsOf(X) == CASE X = "A" -> 1..3 [] X = "B" -> 4..6 [] X = "C" -> 7..9
VecStep(X) ==
    /\ vecV' = vecV \cup {X}
    /\ Access(UNION {{AccVec(VecSubs[v][1] + x) : x \in 0..(VecSubs[v][2] - 1)}
                        : v \in VecSubsOf(X)}, {})
    /\ UNCHANGED <<sfA, sfB, ra, rb, w, cw, cval, aDirty, bDirty>>

\* K1 (one launch) and K1a: scan row ī of A_λ (or column j̄ of B_λ), write the
\* flag and σ (K1: tag = flag) or γ (K1a: tag = 2 + flag) into sA/fA.
ScanStep(t, gammaOnly) ==
    LET lam == t[2]  x == t[3]  base == IF gammaOnly THEN 2 ELSE 0 IN
    IF t[1] = "a"
    THEN /\ sfA' = [sfA EXCEPT ![<<lam, x>>] = base + IF flagA[<<lam, x>>] THEN 1 ELSE 0]
         /\ Access(VecA(lam, x) \cup {AccSA(lam, x), AccFA(lam, x)},
                   ReadAB \cup Stale("A" \notin vecV) \cup ClobA(lam, x))
         /\ UNCHANGED <<vecV, sfB, ra, rb, w, cw, cval, aDirty, bDirty>>
    ELSE /\ sfB' = [sfB EXCEPT ![<<lam, x>>] = base + IF flagB[<<lam, x>>] THEN 1 ELSE 0]
         /\ Access(VecB(lam, x) \cup {AccSB(lam, x), AccFB(lam, x)},
                   ReadAB \cup Stale("B" \notin vecV) \cup ClobB(lam, x))
         /\ UNCHANGED <<vecV, sfA, ra, rb, w, cw, cval, aDirty, bDirty>>

\* K1b: re-read row ī of A_λ (or column j̄ of B_λ) and its γ from sA (sB),
\* write σ over γ (the flag is unchanged). Reading anything but γ (junk, or
\* σ already written over it) is stale.
SigmaStep(t) ==
    LET lam == t[2]  x == t[3] IN
    IF t[1] = "a"
    THEN /\ sfA' = [sfA EXCEPT ![<<lam, x>>] = IF FlagOf(@) THEN 1 ELSE 0]
         /\ Access(VecA(lam, x) \cup {AccSA(lam, x)},
                   ReadAB \cup ClobA(lam, x)
                   \cup Stale("A" \notin vecV \/ sfA[<<lam, x>>] \notin {2, 3}))
         /\ UNCHANGED <<vecV, sfB, ra, rb, w, cw, cval, aDirty, bDirty>>
    ELSE /\ sfB' = [sfB EXCEPT ![<<lam, x>>] = IF FlagOf(@) THEN 1 ELSE 0]
         /\ Access(VecB(lam, x) \cup {AccSB(lam, x)},
                   ReadAB \cup ClobB(lam, x)
                   \cup Stale("B" \notin vecV \/ sfB[<<lam, x>>] \notin {2, 3}))
         /\ UNCHANGED <<vecV, sfA, ra, rb, w, cw, cval, aDirty, bDirty>>

\* Positions a pack writes: real ones (h < k, residues) and padding (zeros).
RealHs == {h \in KHs : h < k}
PadHs  == {h \in KHs : h >= k}

\* The residue / padding writes of one pack CTA for A row (λ, ī) or B column
\* (λ, ĵ) over the slots `slots`; slot s receives modulus ModOf(gs(s), s),
\* i.e. the residues of group gs(s).
PadA(gi) == IF Bug = "noPad" \/ (Bug = "padMinimal" /\ gi > 1) THEN {} ELSE PadHs
RAWrite(lam, i, slots, gs(_)) ==
    [x \in RADom |->
       IF x[2] = lam /\ x[3] = i /\ x[1] \in slots
       THEN (IF x[4] \in RealHs THEN ModOf(gs(x[1]), x[1])
             ELSE IF x[4] \in PadA(gs(x[1])) THEN Zero ELSE ra[x])
       ELSE ra[x]]
RACells(lam, i, slots, gs(_)) ==
    {AccRA(s, lam, i, h) : s \in slots, h \in RealHs}
    \cup UNION {{AccRA(s, lam, i, h) : h \in PadA(gs(s))} : s \in slots}
BRes(j) == IF j < n THEN RealHs ELSE {}                               \* residues
BPad(j) == IF Bug \in {"noPad", "padMinimal"} THEN {}
           ELSE IF j < n THEN PadHs ELSE KHs                           \* zeros
RBWrite(lam, j, slots, gs(_)) ==
    [x \in RBDom |->
       IF x[2] = lam /\ x[3] = j /\ x[1] \in slots
       THEN (IF x[4] \in BRes(j) THEN ModOf(gs(x[1]), x[1])
             ELSE IF x[4] \in BPad(j) THEN Zero ELSE rb[x])
       ELSE rb[x]]
RBCells(lam, j, slots) == {AccRB(s, lam, j, h) : s \in slots, h \in BRes(j) \cup BPad(j)}

\* K2: quantize row ī of A_λ with σ, write the residues of the moduli of
\* part t[3] of group gi into their slots, K padded with zeros up to k̂.
PackAStep(gi, t) ==
    LET lam == t[1]  i == t[2]  slots == PartSlots(gi, t[3])
        G(s) == gi
    IN
    /\ ra' = RAWrite(lam, i, slots, G)
    /\ Access(VecA(lam, i) \cup {AccSA(lam, i)} \cup RACells(lam, i, slots, G),
              ReadAB \cup Reuse(gi) \cup ClobA(lam, i)
              \cup Stale("A" \notin vecV) \cup SigmaRead(sfA[<<lam, i>>]))
    /\ UNCHANGED <<vecV, sfA, sfB, rb, w, cw, cval, aDirty, bDirty>>

\* K3: same for column j̄ of B_λ; CTAs j̄ ∈ n..n̂-1 write zero rows.
PackBStep(gi, t) ==
    LET lam == t[1]  j == t[2]  slots == PartSlots(gi, t[3])  real == j < n
        G(s) == gi
    IN
    /\ rb' = RBWrite(lam, j, slots, G)
    /\ Access(RBCells(lam, j, slots)
              \cup (IF real THEN VecB(lam, j) \cup {AccSB(lam, j)} ELSE {}),
              Reuse(gi)
              \cup (IF real THEN ReadAB \cup ClobB(lam, j)
                                 \cup Stale("B" \notin vecV) \cup SigmaRead(sfB[<<lam, j>>])
                    ELSE {}))
    /\ UNCHANGED <<vecV, sfA, sfB, ra, w, cw, cval, aDirty, bDirty>>

\* v1.4 KF (K0 + K1 + group 1's packs in one launch), one atomic step per CTA:
\*   <<"v", X>>         writes tensor X's scatter vectors (K0's job);
\*   <<"a", λ, ī, y>>   reads row ī of A_λ through offsets computed from the
\*                      descriptor (no vec read), computes σ and the flag
\*                      itself, packs the slots of part y of group 1 and
\*                      (part 0 only) stores σ / the flag for later kernels;
\*   <<"b", λ, ĵ, y>>   the same for column ĵ of B_λ (ĵ ≥ n: zero rows).
\* Negatives: fusedReadsVec (reads the vec region of its own launch),
\* fusedSharedSigma (parts y > 0 read part 0's σ from sA), fusedNoSigma (σ
\* not stored), fusedNoVec (no vec CTAs), fusedAllGroups (every group packed
\* into the same g slots by the one launch).
FusedSlots(y) == IF Bug = "fusedAllGroups"
                 THEN UNION {PartSlots(gi, y) : gi \in 1..NG} ELSE PartSlots(1, y)
\* The group whose residues a slot ends up holding (the last one written).
FusedGroup(s) == IF Bug \in {"fusedAllGroups", "largeAllGroups"}
                 THEN CHOOSE gi \in 1..NG : s < GrpSize(gi) /\ \A g2 \in (gi + 1)..NG : s >= GrpSize(g2)
                 ELSE 1
FusedStoresSigma(y) == y = 0 /\ Bug # "fusedNoSigma"
FusedStep(t) ==
    IF t[1] = "v" THEN VecStep(t[2])
    ELSE
    LET lam == t[2]  x == t[3]  y == t[4]  slots == FusedSlots(y)
        shared == Bug = "fusedSharedSigma" /\ y > 0      \* σ read from sA / sB
        viaVec == Bug = "fusedReadsVec"
        real   == t[1] = "a" \/ x < n
    IN
    IF t[1] = "a"
    THEN /\ ra' = RAWrite(lam, x, slots, FusedGroup)
         /\ sfA' = IF FusedStoresSigma(y)
                   THEN [sfA EXCEPT ![<<lam, x>>] = IF flagA[<<lam, x>>] THEN 1 ELSE 0]
                   ELSE sfA
         /\ Access(RACells(lam, x, slots, FusedGroup)
                   \cup (IF viaVec THEN VecA(lam, x) ELSE {})
                   \cup (IF FusedStoresSigma(y) \/ shared THEN {AccSA(lam, x)} ELSE {})
                   \cup (IF FusedStoresSigma(y) THEN {AccFA(lam, x)} ELSE {}),
                   ReadAB \cup ClobA(lam, x)
                   \cup Stale(viaVec /\ "A" \notin vecV)
                   \cup (IF shared THEN SigmaRead(sfA[<<lam, x>>]) ELSE {}))
         /\ UNCHANGED <<vecV, sfB, rb, w, cw, cval, aDirty, bDirty>>
    ELSE /\ rb' = RBWrite(lam, x, slots, FusedGroup)
         /\ sfB' = IF real /\ FusedStoresSigma(y)
                   THEN [sfB EXCEPT ![<<lam, x>>] = IF flagB[<<lam, x>>] THEN 1 ELSE 0]
                   ELSE sfB
         /\ Access(RBCells(lam, x, slots)
                   \cup (IF real /\ viaVec THEN VecB(lam, x) ELSE {})
                   \cup (IF real /\ (FusedStoresSigma(y) \/ shared) THEN {AccSB(lam, x)} ELSE {})
                   \cup (IF real /\ FusedStoresSigma(y) THEN {AccFB(lam, x)} ELSE {}),
                   IF real
                   THEN ReadAB \cup ClobB(lam, x)
                        \cup Stale(viaVec /\ "B" \notin vecV)
                        \cup (IF shared THEN SigmaRead(sfB[<<lam, x>>]) ELSE {})
                   ELSE {})
         /\ UNCHANGED <<vecV, sfA, ra, w, cw, cval, aDirty, bDirty>>

\* v1.6 KL (K1 + group 1's pack of one operand in one launch), one atomic
\* step per CTA <<"a", λ, ī, seg>> (or <<"b", λ, ĵ, seg>>): read row ī of A_λ
\* through the scatter vectors (K0 retired before this launch), compute σ
\* and the flag from the whole row (segment 0, the row's only CTA), write the
\* residues of every slot of group 1 and store σ / the flag for the later
\* groups' packs and K5. B columns ĵ ≥ n write zero rows only.
\* Negatives: largeWithK0 (the vectors are read in the launch that writes
\* them), largeRowSplit (segment 1 of a row takes σ from sA, stored by
\* segment 0 in the same launch), largeNoSigma (σ not stored), largeAllGroups
\* (every group of the operand packed into the same g slots by the one launch).
LargeStores(seg) == seg = 0 /\ Bug # "largeNoSigma"
LargeSlots == IF Bug = "largeAllGroups" THEN UNION {0..(GrpSize(gi) - 1) : gi \in 1..NG}
              ELSE 0..(GrpSize(1) - 1)
LargeStep(t) ==
    LET lam == t[2]  x == t[3]  seg == t[4]
        shared == seg # 0                      \* σ taken from sA / sB
        real   == t[1] = "a" \/ x < n
    IN
    IF t[1] = "a"
    THEN /\ ra' = RAWrite(lam, x, LargeSlots, FusedGroup)
         /\ sfA' = IF LargeStores(seg)
                   THEN [sfA EXCEPT ![<<lam, x>>] = IF flagA[<<lam, x>>] THEN 1 ELSE 0]
                   ELSE sfA
         /\ Access(VecA(lam, x) \cup RACells(lam, x, LargeSlots, FusedGroup)
                   \cup (IF LargeStores(seg) \/ shared THEN {AccSA(lam, x)} ELSE {})
                   \cup (IF LargeStores(seg) THEN {AccFA(lam, x)} ELSE {}),
                   ReadAB \cup ClobA(lam, x) \cup Stale("A" \notin vecV)
                   \cup (IF shared THEN SigmaRead(sfA[<<lam, x>>]) ELSE {}))
         /\ UNCHANGED <<vecV, sfB, rb, w, cw, cval, aDirty, bDirty>>
    ELSE /\ rb' = RBWrite(lam, x, LargeSlots, FusedGroup)
         /\ sfB' = IF real /\ LargeStores(seg)
                   THEN [sfB EXCEPT ![<<lam, x>>] = IF flagB[<<lam, x>>] THEN 1 ELSE 0]
                   ELSE sfB
         /\ Access(RBCells(lam, x, LargeSlots)
                   \cup (IF real THEN VecB(lam, x) ELSE {})
                   \cup (IF real /\ (LargeStores(seg) \/ shared) THEN {AccSB(lam, x)} ELSE {})
                   \cup (IF real /\ LargeStores(seg) THEN {AccFB(lam, x)} ELSE {}),
                   IF real
                   THEN ReadAB \cup ClobB(lam, x) \cup Stale("B" \notin vecV)
                        \cup (IF shared THEN SigmaRead(sfB[<<lam, x>>]) ELSE {})
                   ELSE {})
         /\ UNCHANGED <<vecV, sfA, ra, w, cw, cval, aDirty, bDirty>>

\* K4: chunk c of the int8 GEMM for batch z = s·b + λ (modulus l), row ī,
\* the columns of N tile y of y part kr.q; epilogue store-mod (c = 0) or
\* add-mod (c > 0).
\*   stale  a real product reads a non-residue / wrong-modulus operand, a
\*          K-padding product has no zero operand, or add-mod reads a junk W
\*   plane  add-mod folds a K position that the plane already holds (j < n)
\*   dup    a W cell (any column j < n̂) written twice for the same chunk:
\*          store-mod over a written cell, or add-mod re-folding positions
\* W columns j ≥ n are computed (and stored, AccessesSafe) but never read
\* by K5; their tag records the writes only.
\* D1: what byte rel of packed operand X's res region holds, decoded by the
\* §6b.4 layout: <<l, λ, ρ>> = the residue of row (λ, ρ) of X for modulus l
\* (real positions h < k), PZero (K padding, the tail), PJunk (not yet
\* written by imd_pack, or past the region).
PZero == <<-1, -1, -1>>
PJunk == <<0, 0, 0>>
PContent(X, rel) ==
    LET r == PR(X)  pl == b * r * KHat IN
    IF rel < 0 THEN PJunk
    ELSE IF rel >= N * pl
         THEN (IF rel < N * pl + PTail(X) /\ pw[X][PTailRow] THEN PZero ELSE PJunk)
    ELSE LET l == rel \div pl + 1  q == rel % pl  row == q \div KHat  h == q % KHat
             lam == row \div r  rho == row % r
         IN  IF ~pw[X][<<lam, rho>>] THEN PJunk
             ELSE IF h < k THEN <<l, lam, rho>> ELSE PZero
\* K4 / K5 read operand X from its packed buffer (else from the workspace;
\* "packedReadsWs" / "packedSigmaWs": from the absent workspace regions).
K4Pack(X) == X \in Packed /\ Bug # "packedReadsWs"
K5Pack(X) == X \in Packed /\ Bug # "packedSigmaWs"

GemmStep(kr, t) ==
    LET z == t[1]  i == t[2]  s == z \div b  lam == z % b
        l == ModOf(kr.gi, s)  c == kr.c  hs == Chunk(c)
        Js == TileCols(kr.q, t[3])
        add == c > 0 /\ Bug # "storeMod"
        pkA == K4Pack("A")  pkB == K4Pack("B")
        PA(h)    == PContent("A", PK4Rel("A", l, lam, i, h))
        PB(j, h) == PContent("B", PK4Rel("B", l, lam, j, h))
        \* the operand cell holds the residue of THIS row for modulus l / a zero
        RealA(h)    == IF pkA THEN PA(h) = <<l, lam, i>> ELSE ra[<<s, lam, i, h>>] = l
        ZeroA(h)    == IF pkA THEN PA(h) = PZero ELSE ra[<<s, lam, i, h>>] = Zero
        RealB(j, h) == IF pkB THEN PB(j, h) = <<l, lam, j>> ELSE rb[<<s, lam, j, h>>] = l
        ZeroB(j, h) == IF pkB THEN PB(j, h) = PZero ELSE rb[<<s, lam, j, h>>] = Zero
        badIn == \E j \in Js \cap Ns, h \in hs :
                   IF h < k THEN ~RealA(h) \/ ~RealB(j, h)
                   ELSE ~ZeroA(h) /\ ~ZeroB(j, h)
        accA == IF pkA THEN {AccPRes("A", PK4Rel("A", l, lam, i, h)) : h \in hs}
                ELSE {AccRA(s, lam, i, h) : h \in hs}
        accB == IF pkB THEN {AccPRes("B", PK4Rel("B", l, lam, j, h)) : j \in Js, h \in hs}
                ELSE {AccRB(s, lam, j, h) : j \in Js, h \in hs}
        badW  == add /\ \E j \in Js \cap Ns : -1 \in w[<<l, lam, i, j>>]
        dbl   == add /\ \E j \in Js \cap Ns : w[<<l, lam, i, j>>] \cap hs # {}
        dup   == \E j \in Js : IF add THEN w[<<l, lam, i, j>>] \cap hs # {}
                               ELSE w[<<l, lam, i, j>>] # {-1}
    IN
    /\ w' = [x \in WDom |->
               IF x[1] = l /\ x[2] = lam /\ x[3] = i /\ x[4] \in Js
               THEN (IF add THEN w[x] \cup hs ELSE hs)
               ELSE w[x]]
    /\ Access(accA \cup accB \cup {AccW(l, lam, i, j) : j \in Js},
              Stale(badIn \/ badW) \cup (IF dbl THEN {"plane"} ELSE {})
              \cup (IF dup THEN {"dup"} ELSE {}))
    /\ UNCHANGED <<vecV, sfA, sfB, ra, rb, cw, cval, aDirty, bDirty>>

\* K5 (and the DIRECT kernel) for C element e = (λ, ī, j̄). Writing C[e]
\* overwrites A row (λ,ī) / B column (λ,j̄) when C aliases A / B.
WriteC(e, tag) ==
    /\ cval' = [cval EXCEPT ![e] = tag]
    /\ cw'   = [cw EXCEPT ![e] = @ + 1]
    /\ aDirty' = IF Bug = "aliasCA" THEN [aDirty EXCEPT ![<<e[1], e[2]>>] = TRUE]
                 ELSE aDirty
    /\ bDirty' = IF Bug = "aliasCB" THEN [bDirty EXCEPT ![<<e[1], e[3]>>] = TRUE]
                 ELSE bDirty
    /\ UNCHANGED <<vecV, sfA, sfB, ra, rb, w>>

TrivialStep(e) ==
    /\ WriteC(e, IF beta0 THEN "zero" ELSE "betaC")
    /\ Access(VecC(e[1], e[2], e[3]), CReadErr(e) \cup Stale("C" \notin vecV))

\* σ / flag of A row (λ, ī) and B column (λ, j̄) as K5 reads them: from the
\* workspace (sA/fA, sB/fB) or, for a packed operand, from its packed buffer
\* (the flag imd_pack computed from the same X values: the input flag).
K5FlagA(lam, i) == IF K5Pack("A") THEN flagA[<<lam, i>>] ELSE FlagOf(sfA[<<lam, i>>])
K5FlagB(lam, j) == IF K5Pack("B") THEN flagB[<<lam, j>>] ELSE FlagOf(sfB[<<lam, j>>])
K5JunkA(lam, i) == IF K5Pack("A") THEN ~pw["A"][<<lam, i>>] ELSE sfA[<<lam, i>>] = -1
K5JunkB(lam, j) == IF K5Pack("B") THEN ~pw["B"][<<lam, j>>] ELSE sfB[<<lam, j>>] = -1
K5GammaA(lam, i) == ~K5Pack("A") /\ sfA[<<lam, i>>] \in {2, 3}
K5GammaB(lam, j) == ~K5Pack("B") /\ sfB[<<lam, j>>] \in {2, 3}
K5AccFA(lam, i) == IF K5Pack("A") THEN AccPFlag("A", lam, i) ELSE AccFA(lam, i)
K5AccFB(lam, j) == IF K5Pack("B") THEN AccPFlag("B", lam, j) ELSE AccFB(lam, j)
K5AccSA(lam, i) == IF K5Pack("A") THEN AccPSig("A", lam, i) ELSE AccSA(lam, i)
K5AccSB(lam, j) == IF K5Pack("B") THEN AccPSig("B", lam, j) ELSE AccSB(lam, j)

CrtStep(e) ==
    LET lam == e[1]  i == e[2]  j == e[3]
        flag == K5FlagA(lam, i) \/ (Bug # "ignoreFB" /\ K5FlagB(lam, j))
        acc  == VecC(lam, i, j) \cup {K5AccFA(lam, i), K5AccFB(lam, j)}
                \cup (IF flag THEN VecA(lam, i) \cup VecB(lam, j)
                      ELSE {K5AccSA(lam, i), K5AccSB(lam, j)} \cup {AccW(l, lam, i, j) : l \in Ls})
    IN
    /\ WriteC(e, IF flag THEN "native" ELSE "crt")
    /\ Access(acc,
              CReadErr(e)
              \cup Stale("C" \notin vecV \/ K5JunkA(lam, i) \/ K5JunkB(lam, j))
              \cup (IF flag THEN ReadAB \cup ClobA(lam, i) \cup ClobB(lam, j)
                                 \cup Stale("A" \notin vecV \/ "B" \notin vecV)
                    ELSE (IF K5GammaA(lam, i) \/ K5GammaB(lam, j)
                          THEN {"gamma"} ELSE {})
                         \cup (IF \E l \in Ls : w[<<l, lam, i, j>>] # KHs THEN {"plane"}
                               ELSE {})))

DirectStep(e) ==
    LET lam == e[1]  i == e[2]  j == e[3] IN
    /\ WriteC(e, "native")
    /\ Access(VecA(lam, i) \cup VecB(lam, j) \cup VecC(lam, i, j),
              CReadErr(e) \cup ReadAB \cup Stale(vecV # {"A", "B", "C"})
              \cup ClobA(lam, i) \cup ClobB(lam, j))

Body(kr, t) ==
    CASE kr.kind = "vec"    -> VecStep(t)
      [] kr.kind = "scale"  -> ScanStep(t, FALSE)
      [] kr.kind = "gamma"  -> ScanStep(t, TRUE)
      [] kr.kind = "sigma"  -> SigmaStep(t)
      [] kr.kind = "k1merged" -> IF t[1] = "g" THEN ScanStep(Tail(t), TRUE) ELSE SigmaStep(Tail(t))
      [] kr.kind = "packA"  -> PackAStep(kr.gi, t)
      [] kr.kind = "packB"  -> PackBStep(kr.gi, t)
      [] kr.kind = "fused"  -> FusedStep(t)
      [] kr.kind = "scaleop" -> ScanStep(t, FALSE)
      [] kr.kind = "large"  -> LargeStep(t)
      [] kr.kind = "largevec" -> IF t[1] = "v" THEN VecStep(t[2]) ELSE LargeStep(t)
      [] kr.kind = "gemm"   -> GemmStep(kr, t)
      [] kr.kind = "crt"    -> IF TrivialK5 THEN TrivialStep(t) ELSE CrtStep(t)
      [] kr.kind = "direct" -> IF TrivialK5 THEN TrivialStep(t) ELSE DirectStep(t)

-----------------------------------------------------------------------------
(* Streams *)

Live(st)   == pc[st] <= Len(SProg[alpha0][st])
CurIdx(st) == SProg[alpha0][st][pc[st]]
Ready(st)  == Deps[alpha0][CurIdx(st)] \subseteq retired

RunCTA(st, t) ==
    /\ t \notin done[st]
    /\ done' = [done EXCEPT ![st] = @ \cup {t}]
    /\ Body(Prog[alpha0][CurIdx(st)], t)
    /\ UNCHANGED <<pc, retired, inputs>>

\* Kernel boundary: every CTA finished; the next kernel on st may start and
\* the completion event is recorded.
Retire(st) ==
    /\ done[st] = CTATbl[alpha0][CurIdx(st)]
    /\ pc' = [pc EXCEPT ![st] = @ + 1]
    /\ done' = [done EXCEPT ![st] = {}]
    /\ retired' = retired \cup {CurIdx(st)}
    /\ UNCHANGED <<inputs, aDirty, bDirty, vecV, sfA, sfB, ra, rb, w, cw, cval,
                   touched, err>>

AllDone == \A st \in Streams : ~Live(st)

Next ==
    \/ \E st \in Streams :
          /\ Live(st) /\ Ready(st)
          /\ \/ \E t \in CTATbl[alpha0][CurIdx(st)] : RunCTA(st, t)
             \/ Retire(st)
    \/ PackWrite                        \* only with "packInFlight"
    \/ (AllDone /\ UNCHANGED vars)      \* legal termination (deadlock check stays on)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

-----------------------------------------------------------------------------
(* What C[e] must end up holding (§3.7), independent of g, K-chunking,     *)
(* launch split and stream assignment.                                     *)
Expected(e) ==
    IF alpha0 \/ k = 0 THEN (IF beta0 THEN "zero" ELSE "betaC")
    ELSE IF UseDirect \/ flagA[<<e[1], e[2]>>] \/ flagB[<<e[1], e[3]>>] THEN "native"
    ELSE "crt"

-----------------------------------------------------------------------------
(* Invariants and properties *)

TypeOK ==
    /\ vecV \subseteq {"A", "B", "C"}
    /\ sfA \in [RowsA -> -1..3] /\ sfB \in [ColsB -> -1..3]
    /\ ra \in [RADom -> Zero..N] /\ rb \in [RBDom -> Zero..N]
    /\ \A x \in WDom : w[x] \subseteq {-1} \cup KHs
    /\ aDirty \in [RowsA -> BOOLEAN] /\ bDirty \in [ColsB -> BOOLEAN]
    /\ cw \in [Elems -> Nat] /\ cval \in [Elems -> CTags]
    /\ touched \subseteq Active \cup {"pvec"} \cup PackRegs
    /\ err \subseteq ErrKinds
    /\ DOMAIN pw = Packed
    /\ \A X \in Packed : pw[X] \in [PRows(X) \cup {PTailRow} -> BOOLEAN]

\* §4 "No stale reads": every read (vec, σ/flags, RA/RB chunk slices,
\* W before add-mod, C before fma) sees the value written for the current
\* call / group / chunk, never junk or a later group's residues; a K-padding
\* product always has a zero operand.
NoStaleRead == "stale" \notin err

\* §4 K1 split: σ/τ are final before any pack or K5 reads them (sA/sB hold
\* K1a's γ until K1b retires; nothing but K1b reads γ).
SigmaFinalAtUse == "gamma" \notin err

\* §4 "Residue buffers are reused across groups only after their GEMMs
\* retire": no pack CTA of group G+1 runs while a GEMM of group G is unretired.
ReuseAfterRetire == "reuse" \notin err

\* §4 "Exactly once ... after all N planes for it are final", §3.5 "Final
\* W_l ≡ (A'B') mod p_l": every W plane K5 reads for a CRT element has
\* folded every K position of [0, k̂) exactly once (store-mod first, then
\* add-mod; no position twice, none missing).
PlanesFinalAtCRT == "plane" \notin err

\* §4 launch limits, the n̂ split: no GEMM launch writes a W cell (any
\* column j < n̂, incl. the padding columns) that an earlier launch of the
\* same K-chunk already wrote (the y parts and the N tiles inside each
\* launch are disjoint) ...
PlanesWrittenOnce == "dup" \notin err
\* ... and together they cover it: after a complete emulated run every
\* cell of every plane (every column of [0, n̂)) has folded exactly the
\* K positions [0, k̂).
PlanesCoverAtEnd ==
    (AllDone /\ FullWs /\ ~alpha0) => \A x \in WDom : w[x] = KHs

\* §4 Exactly once.
FinalAtMostOnce       == \A e \in Elems : cw[e] <= 1
FinalExactlyOnceAtEnd == AllDone => \A e \in Elems : cw[e] = 1
Termination           == <>AllDone

\* §3.1/§3.7: flagged elements take the native path, unflagged the CRT path;
\* α = 0 or k = 0 gives β·C (or 0). Per-element refinement of the abstract
\* contraction: the written value is Expected(e), a function of the inputs
\* only (not of g, chunking, the z or y launch splits, the N tile size,
\* stream, or interleaving). With PlanesFinalAtCRT (every plane read by K5
\* has folded exactly [0, k̂)) this is "the result is independent of the
\* split": the models with and without the n̂ split agree element-wise.
CorrectPath == \A e \in Elems : cw[e] >= 1 => cval[e] = Expected(e)

\* §3.7 read rules.
BetaZeroNoCRead   == "readC" \notin err
AlphaZeroNoABRead == "readAB" \notin err

\* §4 "C may not alias A or B": no kernel reads an A row or B column that C
\* has overwritten (K5's native path reads A and B while other K5 CTAs write C).
NoInputClobber == "clobber" \notin err

\* §4a bounded workspace: every byte touched lies in its own region, in no
\* other region, and below ws_bytes.
AccessesSafe == "oob" \notin err

\* §4a tightness: a complete emulated run (α ≠ 0) touches every non-empty
\* region the formula allocates (with LayoutExact: no byte is allocated for
\* nothing except alignment padding).
\* D1: a complete emulated run reads every packed buffer's residues and
\* flags, and its σ when some output takes the CRT path.
RegionsUsed ==
    (AllDone /\ ~alpha0) =>
        /\ {r \in Active : Bytes(r) > 0} \subseteq touched
        /\ FullWs =>
             \A X \in Packed : PackBytes(X) > 0 =>
               /\ {PSub(X, "res"), PSub(X, "flag")} \subseteq touched
               /\ (\E e \in Elems : ~flagA[<<e[1], e[2]>>] /\ ~flagB[<<e[1], e[3]>>])
                    => PSub(X, "sig") \in touched

\* v1.2 resident vectors: the plan's device buffer (what imd_plan_device_bytes
\* reports) is exactly the nine vectors, gap-free (8 bytes per entry), and
\* none for a generate plan. Constant-level.
PlanBytesExact ==
    /\ ~VecResident => PlanBytes = 0
    /\ (VecResident /\ ~Empty) =>
          /\ NCells("vec") * 8 = PlanBytes
          /\ ByteSet("vec") = 0..(PlanBytes - 1)

\* §4a layout facts. These are constant-level: TLC evaluates them once before
\* the search and reports a failure as "The invariant of X is equal to FALSE".
RegionsDisjoint ==
    \A r1, r2 \in Active : r1 # r2 =>
        \/ Bytes(r1) = 0 \/ Bytes(r2) = 0
        \/ Off(r1) + Bytes(r1) <= Off(r2) \/ Off(r2) + Bytes(r2) <= Off(r1)
RegionsAligned == \A r \in Active : Off(r) % AlignWS = 0 /\ Off(r) % Width(r) = 0
RegionsFit     == \A r \in Active : Off(r) + Bytes(r) <= WorkspaceSize

\* D1 §6b.4 (constant-level): each packed sub-region's index map is a
\* bijection onto [off, off + bytes): σ and flags per row (λ, ρ), the
\* residues of N planes × b·r rows × k̂ positions, then the tail; the
\* sub-regions are aligned, in order, and pack_bytes is their aligned sum.
PCells(X, s) ==
    CASE s = "sig"  -> {AccPSig(X, x[1], x[2]) : x \in PRows(X)}
      [] s = "flag" -> {AccPFlag(X, x[1], x[2]) : x \in PRows(X)}
      [] s = "res"  -> {AccPRes(X, PResRel(X, l, x[1], x[2], h)) : l \in Ls, x \in PRows(X), h \in KHs}
                       \cup {AccPRes(X, N * b * PR(X) * KHat + u) : u \in 0..(PTail(X) - 1)}
PNCells(X, s) == CASE s = "sig" -> b * PR(X) [] s = "flag" -> b * PR(X)
                   [] s = "res" -> N * b * PR(X) * KHat + PTail(X)
PWidth(s) == IF s = "sig" THEN 2 ELSE 1
PackLayoutExact ==
    \A X \in Packed :
        /\ \A s \in {"sig", "flag", "res"} :
              /\ PNCells(X, s) * PWidth(s) = PSubBytes(X, s)
              /\ UNION {x[2]..(x[2] + x[3] - 1) : x \in PCells(X, s)}
                   = PSubOff(X, s)..(PSubOff(X, s) + PSubBytes(X, s) - 1)
              /\ PSubOff(X, s) % AlignWS = 0
        /\ PSubOff(X, "flag") >= PSubOff(X, "sig") + PSubBytes(X, "sig")
        /\ PSubOff(X, "res") >= PSubOff(X, "flag") + PSubBytes(X, "flag")
        /\ PackBytes(X) > 0 => PackBytes(X) = PSubOff(X, "res") + Align(PSubBytes(X, "res"))

\* §4a region sizes and in-region index maps are exact: each region's
\* formula term equals (#logical cells)·(element width), and the cells' byte
\* ranges are exactly [off, off + bytes). Together these say the address map
\* is injective (no two logical elements share a byte: no aliasing inside a
\* region) and gap-free, which is what makes the separate-buffer abstraction
\* of this module sound. Covers the nine vec sub-vectors too, and (D1) the
\* packed buffers' sub-regions.
LayoutExact ==
    /\ \A r \in Active :
          /\ NCells(r) * Width(r) = Bytes(r)
          /\ ByteSet(r) = Off(r)..(Off(r) + Bytes(r) - 1)
    /\ PackLayoutExact

\* §3.5 K-chunks: non-empty, contiguous, at most K_c' long (the int32
\* exactness bound is per chunk), pairwise disjoint and covering [0, k̂).
ChunksPartition ==
    /\ UNION {Chunk(c) : c \in AllChunks} = KHs
    /\ \A c1, c2 \in AllChunks : c1 # c2 => Chunk(c1) \cap Chunk(c2) = {}
    /\ \A c \in AllChunks :
          LET C == Chunk(c) IN
          /\ C # {}
          /\ Cardinality(C) <= ChunkLen
          /\ \E lo \in C : C = lo..(lo + Cardinality(C) - 1)

\* §4 GEMM configuration: every batched-GEMM operand pointer (per batch and
\* per K-chunk), leading dimension and batch stride is a multiple of the
\* CUTLASS alignment (16 int8 / u8 elements, scaled to AlignK / AlignN).
GemmAligned ==
    FullWs =>
      /\ KHat % AlignK = 0 /\ NHat % AlignN = 0
      /\ (m * KHat) % AlignK = 0 /\ (NHat * KHat) % AlignK = 0 /\ (m * NHat) % AlignN = 0
      /\ \A gi \in 1..NG : \A c \in AllChunks, z \in 0..(NZ(gi) - 1) :
           LET s == z \div b  lam == z % b IN
           /\ AddrRA(s, lam, 0, c * ChunkLen) % AlignK = 0
           /\ AddrRB(s, lam, 0, c * ChunkLen) % AlignK = 0
           /\ AddrW(ModOf(gi, s), lam, 0, 0) % AlignN = 0
           \* the y part's B and W pointers (column j0 = YStart(q))
           /\ \A q \in 0..(NYParts - 1) :
                /\ AddrRB(s, lam, YStart(q), c * ChunkLen) % AlignK = 0
                /\ AddrW(ModOf(gi, s), lam, 0, YStart(q)) % AlignN = 0
           \* D1: a packed operand's pointers (per plane, batch, chunk, y part)
           /\ \A X \in Packed : \A q \in 0..(NYParts - 1) :
                LET r0 == IF X = "B" THEN YStart(q) ELSE 0 IN
                (PSubOff(X, "res") + PK4Rel(X, ModOf(gi, s), lam, r0, c * ChunkLen)) % AlignK = 0

\* §4 DIRECT path / §4a: "K-empty -> DIRECT is documented dispatch by
\* descriptor". The DIRECT kernel is launched iff the output is non-empty
\* and (mode = DIRECT or the descriptor is K-empty), for either value of
\* the α = 0 input (a host-side fact that cannot change the path), no
\* emulation kernel runs on that path, and the workspace the §4a formula
\* reports is exactly the layout of the path taken (align(vec) there).
\* Constant-level.
DispatchByDescriptor ==
    /\ \A a \in BOOLEAN :
         LET kinds == {Prog[a][i].kind : i \in DOMAIN Prog[a]} IN
         /\ ("direct" \in kinds) <=> (~Empty /\ (Direct \/ KEmpty))
         /\ "direct" \in kinds => kinds \cap {"scale", "gamma", "sigma", "k1merged", "packA",
                                              "packB", "fused", "scaleop", "large",
                                              "largevec", "gemm", "crt"} = {}
    /\ WorkspaceSize = SumSeq([i \in DOMAIN Regions |-> Align(Bytes(Regions[i]))])

\* §4 launch limits: each batched launch has gridDim.z <= MaxZ, and the
\* launches of a group cover its |G|·b batches exactly once.
LaunchGridOK ==
    FullWs =>
      \A gi \in 1..NG :
        /\ \A p \in 0..(NParts(gi) - 1) : Cardinality(PartZ(gi, p)) \in 1..MaxZ
        /\ UNION {PartZ(gi, p) : p \in 0..(NParts(gi) - 1)} = 0..(NZ(gi) - 1)
        /\ \A p1, p2 \in 0..(NParts(gi) - 1) :
              p1 # p2 => PartZ(gi, p1) \cap PartZ(gi, p2) = {}

\* §4 launch limits, gridDim.y: n̂ is split into y parts, each a launch of
\* GridY(q) ∈ 1..MaxY N tiles; the y parts partition [0, n̂) (no gap, no
\* overlap) and so do the N tiles inside each part, each tile non-empty and
\* at most TileN wide. Constant-level.
LaunchGridYOK ==
    FullWs =>
      /\ \A q \in 0..(NYParts - 1) :
           /\ GridY(q) \in 1..MaxY
           /\ UNION {TileCols(q, y) : y \in 0..(GridY(q) - 1)} = YCols(q)
           /\ \A y \in 0..(GridY(q) - 1) : Cardinality(TileCols(q, y)) \in 1..TileN
      /\ UNION {YCols(q) : q \in 0..(NYParts - 1)} = NHs
      /\ \A q1, q2 \in 0..(NYParts - 1) : q1 # q2 => YCols(q1) \cap YCols(q2) = {}
=============================================================================
