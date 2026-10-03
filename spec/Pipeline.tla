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
AllRegions == IF VecInWs THEN <<"sA", "sB", "fA", "fB", "vec", "RA", "RB", "W">>
              ELSE <<"sA", "sB", "fA", "fB", "RA", "RB", "W">>
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
\* K1 as launched: one kernel (v1.1), or K1a then K1b (K1Split).
K1Seq == CASE ~K1Split          -> <<KScale>>
           [] Bug = "k1Merged"  -> <<Kr("k1merged", 0, 0, 0)>>
           [] Bug = "noSigma"   -> <<KGamma>>
           [] OTHER             -> <<KGamma, KSigma>>
\* The kernel whose retirement makes σ/τ final (the packs' RAW edge, 2 streams).
K1FinalKind == CASE ~K1Split               -> "scale"
                 [] Bug = "packWaitsGamma" -> "gamma"
                 [] OTHER                  -> "sigma"
KCrt    == Kr("crt", 0, 0, 0)
KDirect == Kr("direct", 0, 0, 0)

\* For chunk c, for z part p, for y part q: one batched GEMM launch.
GemmSeq(gi) == LET P == NParts(gi)  Q == NYParts IN
               [i \in 1..(NCh * P * Q) |->
                  KrQ("gemm", gi, (i - 1) \div (P * Q), ((i - 1) \div Q) % P, (i - 1) % Q)]
RECURSIVE GroupsFrom(_)
GroupsFrom(gi) == IF gi > NG THEN <<>>
                  ELSE <<Kr("packA", gi, 0, 0), Kr("packB", gi, 0, 0)>>
                       \o GemmSeq(gi) \o GroupsFrom(gi + 1)
GP == GroupsFrom(1)

\* K0 runs unless the vectors are resident in the plan.
K0Seq == IF VecResident THEN <<>> ELSE <<KVec>>
ProgFull == IF Empty THEN <<>>
            ELSE IF UseDirect THEN K0Seq \o <<KDirect>>
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
K1CTAs == {<<"a", x[1], x[2]>> : x \in RowsA} \cup {<<"b", x[1], x[2]>> : x \in ColsB}
CTAsOf(kr) ==
    CASE kr.kind = "vec"    -> K0CTAs                 \* one per tensor's vectors
      [] kr.kind \in {"scale", "gamma"} -> K1CTAs     \* one per A row / B column
      [] kr.kind = "sigma"  -> IF Bug = "sigmaRowSplit"
                               THEN {<<t[1], t[2], t[3], hf>> : t \in K1CTAs, hf \in {0, 1}}
                               ELSE K1CTAs
      [] kr.kind = "k1merged" -> {<<"g">> \o t : t \in K1CTAs} \cup {<<"s">> \o t : t \in K1CTAs}
      [] kr.kind = "packA"  -> RowsA                    \* one A row each
      [] kr.kind = "packB"  -> Bs \X NHs                \* one (padded) B column each
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
    err        \* ghost: kinds of violations observed

vars    == <<pc, done, retired, alpha0, beta0, flagA, flagB, aDirty, bDirty,
             vecV, sfA, sfB, ra, rb, w, cw, cval, touched, err>>
inputs  == <<alpha0, beta0, flagA, flagB>>

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

\* K2: quantize row ī of A_λ with σ, write the residues of every modulus of
\* group gi into slot s, K padded with zeros up to k̂.
PackAStep(gi, t) ==
    LET lam == t[1]  i == t[2]  sz == GrpSize(gi)
        pad == IF Bug = "noPad" \/ (Bug = "padMinimal" /\ gi > 1) THEN {} ELSE PadHs
    IN
    /\ ra' = [x \in RADom |->
                IF x[2] = lam /\ x[3] = i /\ x[1] < sz
                THEN (IF x[4] \in RealHs THEN ModOf(gi, x[1])
                      ELSE IF x[4] \in pad THEN Zero ELSE ra[x])
                ELSE ra[x]]
    /\ Access(VecA(lam, i) \cup {AccSA(lam, i)}
              \cup {AccRA(s, lam, i, h) : s \in 0..(sz - 1), h \in RealHs \cup pad},
              ReadAB \cup Reuse(gi) \cup ClobA(lam, i)
              \cup Stale("A" \notin vecV) \cup SigmaRead(sfA[<<lam, i>>]))
    /\ UNCHANGED <<vecV, sfA, sfB, rb, w, cw, cval, aDirty, bDirty>>

\* K3: same for column j̄ of B_λ; CTAs j̄ ∈ n..n̂-1 write zero rows.
PackBStep(gi, t) ==
    LET lam == t[1]  j == t[2]  sz == GrpSize(gi)  real == j < n
        noB == Bug \in {"noPad", "padMinimal"}
        res == IF real THEN RealHs ELSE {}                      \* residues
        pad == IF noB THEN {} ELSE IF real THEN PadHs ELSE KHs  \* zeros
    IN
    /\ rb' = [x \in RBDom |->
                IF x[2] = lam /\ x[3] = j /\ x[1] < sz
                THEN (IF x[4] \in res THEN ModOf(gi, x[1])
                      ELSE IF x[4] \in pad THEN Zero ELSE rb[x])
                ELSE rb[x]]
    /\ Access({AccRB(s, lam, j, h) : s \in 0..(sz - 1), h \in res \cup pad}
              \cup (IF real THEN VecB(lam, j) \cup {AccSB(lam, j)} ELSE {}),
              Reuse(gi)
              \cup (IF real THEN ReadAB \cup ClobB(lam, j)
                                 \cup Stale("B" \notin vecV) \cup SigmaRead(sfB[<<lam, j>>])
                    ELSE {}))
    /\ UNCHANGED <<vecV, sfA, sfB, ra, w, cw, cval, aDirty, bDirty>>

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
GemmStep(kr, t) ==
    LET z == t[1]  i == t[2]  s == z \div b  lam == z % b
        l == ModOf(kr.gi, s)  c == kr.c  hs == Chunk(c)
        Js == TileCols(kr.q, t[3])
        add == c > 0 /\ Bug # "storeMod"
        A(h)    == ra[<<s, lam, i, h>>]
        B(j, h) == rb[<<s, lam, j, h>>]
        badIn == \E j \in Js \cap Ns, h \in hs :
                   IF h < k THEN A(h) # l \/ B(j, h) # l
                   ELSE A(h) # Zero /\ B(j, h) # Zero
        badW  == add /\ \E j \in Js \cap Ns : -1 \in w[<<l, lam, i, j>>]
        dbl   == add /\ \E j \in Js \cap Ns : w[<<l, lam, i, j>>] \cap hs # {}
        dup   == \E j \in Js : IF add THEN w[<<l, lam, i, j>>] \cap hs # {}
                               ELSE w[<<l, lam, i, j>>] # {-1}
    IN
    /\ w' = [x \in WDom |->
               IF x[1] = l /\ x[2] = lam /\ x[3] = i /\ x[4] \in Js
               THEN (IF add THEN w[x] \cup hs ELSE hs)
               ELSE w[x]]
    /\ Access({AccRA(s, lam, i, h) : h \in hs}
              \cup {AccRB(s, lam, j, h) : j \in Js, h \in hs}
              \cup {AccW(l, lam, i, j) : j \in Js},
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

CrtStep(e) ==
    LET lam == e[1]  i == e[2]  j == e[3]
        flag == FlagOf(sfA[<<lam, i>>]) \/ (Bug # "ignoreFB" /\ FlagOf(sfB[<<lam, j>>]))
        acc  == VecC(lam, i, j) \cup {AccFA(lam, i), AccFB(lam, j)}
                \cup (IF flag THEN VecA(lam, i) \cup VecB(lam, j)
                      ELSE {AccSA(lam, i), AccSB(lam, j)} \cup {AccW(l, lam, i, j) : l \in Ls})
    IN
    /\ WriteC(e, IF flag THEN "native" ELSE "crt")
    /\ Access(acc,
              CReadErr(e)
              \cup Stale("C" \notin vecV \/ sfA[<<lam, i>>] = -1 \/ sfB[<<lam, j>>] = -1)
              \cup (IF flag THEN ReadAB \cup ClobA(lam, i) \cup ClobB(lam, j)
                                 \cup Stale("A" \notin vecV \/ "B" \notin vecV)
                    ELSE (IF sfA[<<lam, i>>] \in {2, 3} \/ sfB[<<lam, j>>] \in {2, 3}
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
    /\ touched \subseteq Active \cup {"pvec"}
    /\ err \subseteq ErrKinds

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
RegionsUsed == (AllDone /\ ~alpha0) => {r \in Active : Bytes(r) > 0} \subseteq touched

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

\* §4a region sizes and in-region index maps are exact: each region's
\* formula term equals (#logical cells)·(element width), and the cells' byte
\* ranges are exactly [off, off + bytes). Together these say the address map
\* is injective (no two logical elements share a byte: no aliasing inside a
\* region) and gap-free, which is what makes the separate-buffer abstraction
\* of this module sound. Covers the nine vec sub-vectors too.
LayoutExact ==
    \A r \in Active :
        /\ NCells(r) * Width(r) = Bytes(r)
        /\ ByteSet(r) = Off(r)..(Off(r) + Bytes(r) - 1)

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
                                              "packB", "gemm", "crt"} = {}
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
