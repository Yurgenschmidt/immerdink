---------------------------- MODULE NativePipeline ----------------------------
(***************************************************************************)
(* The launch sequence of one IMD_NATIVE imd_contract call (issue #75,     *)
(* docs/NOTES-NATIVE.md §4.2) and its workspace layout.                    *)
(*                                                                         *)
(* A NATIVE call computes C <- alpha * A B + beta * C with FP64 tensor-core *)
(* GEMMs. Each operand X in {A, B} is either read IN PLACE (it is affine)  *)
(* or PACKED: a gather kernel copies it, K-contiguous, into a workspace    *)
(* buffer first. With k-slabs the contracted index K = 0..K-1 is cut into  *)
(* slabs of L positions (the last one may be shorter); per slab s, in this *)
(* order on one stream:                                                    *)
(*     PACK_A(s)  (if A is packed)   positions of slab s -> bufA[0..L-1]   *)
(*     PACK_B(s)  (if B is packed)   positions of slab s -> bufB[0..L-1]   *)
(*     GEMM(s)    reads the slab's positions (buffer or in place), then    *)
(*                its epilogue writes C:                                   *)
(*                  slab 0:  beta = the caller's beta  (beta = 0: C is NOT *)
(*                           read: "zero" mode)                            *)
(*                  slab >0: beta = 1 (C read, the earlier slabs kept)     *)
(*                  alpha is applied to the slab's own sum, every slab.    *)
(* K0 (the scatter-vector kernel) runs first iff some kernel needs the     *)
(* vectors (a packed operand's gather or a scattered C store) and they are *)
(* not plan-resident (IMD_VEC_RESIDENT).                                   *)
(*                                                                         *)
(* Rows and columns are independent, so ONE row of A, ONE column of B and  *)
(* ONE output element stand for all (as in Pipeline's tags). Values are    *)
(* ghost tags:                                                             *)
(*   bufX[q]   <<s, h>>: position h of slab s; Junk: not written this call *)
(*   c[t]      for each term t of the output (the caller's C0 and every K  *)
(*             position h): n = how many times it is in C, a = how many    *)
(*             times alpha multiplied it, b = how many times beta did.     *)
(*             Correct end state: C0 once with one beta and no alpha (or   *)
(*             absent when beta = 0), every h exactly once with one alpha  *)
(*             and no beta.                                                *)
(* Kernels are launches on streams; a launch may start once every earlier  *)
(* launch of its stream has finished (stream order; no events modelled).   *)
(* Inside a launch, the steps (one per pack cell; one per GEMM read, then   *)
(* the epilogue after all of its reads) interleave freely.                 *)
(*                                                                         *)
(* Workspace (§4a style): regions vec, PA, PB at AlignWS-aligned offsets;  *)
(* a packed operand's region holds b * rows * Lhat FP64 values, Lhat = L   *)
(* rounded up to the GEMM K tile Q (one row here: Lhat cells), reused by   *)
(* every slab. Checked: offsets aligned, regions disjoint, every pack      *)
(* write inside its own region.                                            *)
(*                                                                         *)
(* Out of scope here: alpha = 0 / k = 0 (beta-only kernel, no slabs) and   *)
(* K-empty (DIRECT) dispatch, which are IndexMap's; numerical error, which *)
(* is PROOFS L7's (the slab composition is DESIGN §3.8's K-segmentation).  *)
(* Mut names one realistic bug; every mutant has a model that fails.      *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, Sequences

CONSTANTS K,        \* contracted length (positions 0..K-1)
          L,        \* slab length (k_slab; L >= K means one slab)
          Q,        \* GEMM K tile: the pack buffer width is L rounded up to Q
          PkA, PkB, \* operand packed (TRUE) or read in place (FALSE)
          CScatter, \* C stored through its scatter vectors (not affine)
          VecRes,   \* IMD_VEC_RESIDENT: the vectors are in plan memory, no K0
          Beta0,    \* the caller's beta is 0
          AlignWS,  \* workspace region alignment (bytes; real 256)
          VecBytes, \* bytes of the vec region (abstract)
          Mut       \* "none" or the name of a mutant

ASSUME K \in 1..4 /\ L \in 1..4 /\ Q \in 1..4 /\ AlignWS \in 1..64 /\ VecBytes \in 1..64
ASSUME PkA \in BOOLEAN /\ PkB \in BOOLEAN /\ CScatter \in BOOLEAN
ASSUME VecRes \in BOOLEAN /\ Beta0 \in BOOLEAN

Ops   == {"A", "B"}
Pk(x) == IF x = "A" THEN PkA ELSE PkB
S     == (K + L - 1) \div L                       \* number of slabs
Slabs == 0..(S - 1)
Start(s) == s * L
\* end (exclusive) of slab s: the last slab is shorter
End(s)   == IF Mut = "slab_last_full" THEN Start(s) + L
            ELSE IF Start(s) + L < K THEN Start(s) + L ELSE K
SlabPos(s) == Start(s)..(End(s) - 1)
Lhat  == ((L + Q - 1) \div Q) * Q                 \* pack buffer cells per row
Junk  == <<-1, -1>>

NeedVec == (PkA \/ PkB \/ CScatter)

\* ------------------------------------------------------------ launches --
\* A launch: [kind, slab, op, st (stream)]. The design: one stream.
PackStream == IF Mut = "two_streams" THEN 2 ELSE 1
PackL(s, x) == <<[kind |-> "pack", slab |-> s, op |-> x, st |-> PackStream]>>
GemmL(s)    == <<[kind |-> "gemm", slab |-> s, op |-> "", st |-> 1]>>
PacksOf(s)  == (IF PkA /\ (Mut # "pack_once" \/ s = 0) THEN PackL(s, "A") ELSE << >>)
               \o (IF PkB /\ (Mut # "pack_once" \/ s = 0) THEN PackL(s, "B") ELSE << >>)
RECURSIVE SlabSeq(_)
SlabSeq(s) == IF s >= S THEN << >>
              ELSE PacksOf(s) \o GemmL(s) \o SlabSeq(s + 1)
RECURSIVE PackSeq(_)
PackSeq(s) == IF s >= S THEN << >> ELSE PacksOf(s) \o PackSeq(s + 1)
RECURSIVE GemmSeq(_)
GemmSeq(s) == IF s >= S THEN << >> ELSE GemmL(s) \o GemmSeq(s + 1)
K0L == IF NeedVec /\ ~VecRes /\ Mut # "no_k0"
       THEN <<[kind |-> "k0", slab |-> 0, op |-> "", st |-> 1]>> ELSE << >>
Launches == K0L \o (IF Mut = "pack_all_first" THEN PackSeq(0) \o GemmSeq(0) ELSE SlabSeq(0))
NL == Len(Launches)
LIdx == 1..NL

\* the epilogue's beta mode for slab s
BetaMode(s) ==
    IF s = 0
    THEN (IF Beta0 /\ Mut # "read_c_beta0" THEN "zero" ELSE "caller")
    ELSE (IF Mut = "beta_every_slab" THEN "caller"
          ELSE IF Mut = "later_beta_zero" THEN "zero" ELSE "one")
AlphaHere(s) == Mut # "alpha_last_only" \/ s = S - 1

\* ------------------------------------------------------------ workspace --
AlignUp(x) == ((x + AlignWS - 1) \div AlignWS) * AlignWS
VecSz == IF NeedVec /\ ~VecRes THEN VecBytes ELSE 0
LastLen == K - (S - 1) * L
PkSz(x) == IF ~Pk(x) THEN 0
           ELSE IF Mut = "ws_last_slab" THEN 8 * LastLen ELSE 8 * Lhat
OffVec == 0
OffA   == OffVec + AlignUp(VecSz)
OffB   == OffA + AlignUp(PkSz("A"))
WsBytes == OffB + AlignUp(PkSz("B"))
Off(x) == IF x = "A" THEN OffA ELSE OffB

RegionsOK ==
    /\ \A o \in {OffVec, OffA, OffB} : o % AlignWS = 0
    /\ OffVec + VecSz <= OffA /\ OffA + PkSz("A") <= OffB /\ OffB + PkSz("B") <= WsBytes

\* -------------------------------------------------------------- state --
C0 == -1                                          \* the caller's original C term
Terms == {C0} \cup (0..(K - 1))
Lim == 3                                          \* counts saturate (keeps the state finite)
Sat(v) == IF v >= Lim THEN Lim ELSE v

VARIABLES
  vec,     \* the scatter vectors are valid
  buf,     \* [Ops -> [0..Lhat-1 -> tag]]
  todo,    \* [LIdx -> set of pending steps]
  rd,      \* [LIdx -> set of positions a GEMM launch has read]
  c,       \* [Terms -> [n, a, b]]
  cInit,   \* C still holds the caller's original value (no epilogue yet)
  bad      \* set of violation names

vars == <<vec, buf, todo, rd, c, cInit, bad>>

\* the steps of launch i
StepsOf(i) ==
    LET l == Launches[i] IN
    CASE l.kind = "k0"   -> {<<"v", 0>>}
      [] l.kind = "pack" -> {<<"w", h>> : h \in SlabPos(l.slab)}
      [] l.kind = "gemm" -> {<<"r", h>> : h \in SlabPos(l.slab)} \cup {<<"e", 0>>}

Init ==
  /\ vec = VecRes
  /\ buf = [x \in Ops |-> [q \in 0..(Lhat - 1) |-> Junk]]
  /\ todo = [i \in LIdx |-> StepsOf(i)]
  /\ rd = [i \in LIdx |-> {}]
  /\ c = [t \in Terms |-> IF t = C0 THEN [n |-> 1, a |-> 0, b |-> 0] ELSE [n |-> 0, a |-> 0, b |-> 0]]
  /\ cInit = TRUE
  /\ bad = {}

Finished(i) == todo[i] = {}
\* stream order: every earlier launch of the same stream has finished
MayRun(i) == \A j \in 1..(i - 1) : Launches[j].st = Launches[i].st => Finished(j)

K0Step(i) ==
  /\ Launches[i].kind = "k0" /\ <<"v", 0>> \in todo[i]
  /\ vec' = TRUE
  /\ todo' = [todo EXCEPT ![i] = @ \ {<<"v", 0>>}]
  /\ UNCHANGED <<buf, rd, c, cInit, bad>>

\* a pack writes cell q = h - Start(s) of its operand's buffer, gathering through the vectors
PackStep(i, h) ==
  LET l == Launches[i]  q == h - Start(l.slab)  addr == Off(l.op) + 8 * q IN
  /\ l.kind = "pack" /\ <<"w", h>> \in todo[i]
  /\ buf' = [buf EXCEPT ![l.op] = IF q \in DOMAIN @ THEN [@ EXCEPT ![q] = <<l.slab, h>>] ELSE @]
  /\ bad' = bad \cup (IF ~vec THEN {"pack_reads_junk_vec"} ELSE {})
                \cup (IF addr < Off(l.op) \/ addr + 8 > Off(l.op) + PkSz(l.op) \/ q \notin DOMAIN buf[l.op]
                      THEN {"pack_outside_region"} ELSE {})
                \cup (IF h >= K THEN {"read_past_k"} ELSE {})
  /\ todo' = [todo EXCEPT ![i] = @ \ {<<"w", h>>}]
  /\ UNCHANGED <<vec, rd, c, cInit>>

\* a GEMM reads position h of both operands: from the buffer (must hold <<s, h>>) or in place
OkRead(x, s, h) == ~Pk(x) \/ (h - Start(s) \in DOMAIN buf[x] /\ buf[x][h - Start(s)] = <<s, h>>)
ReadStep(i, h) ==
  LET l == Launches[i] IN
  /\ l.kind = "gemm" /\ <<"r", h>> \in todo[i]
  /\ rd' = [rd EXCEPT ![i] = @ \cup {h}]
  /\ bad' = bad \cup (IF \A x \in Ops : OkRead(x, l.slab, h) THEN {} ELSE {"stale_read"})
                \cup (IF h >= K THEN {"read_past_k"} ELSE {})
  /\ todo' = [todo EXCEPT ![i] = @ \ {<<"r", h>>}]
  /\ UNCHANGED <<vec, buf, c, cInit>>

\* the epilogue, after every read of the launch: C <- RN(alpha * acc) + beta_mode * C
Epi(i) ==
  LET l == Launches[i]  s == l.slab  bm == BetaMode(s)
      old(t) == IF bm = "zero" THEN [n |-> 0, a |-> 0, b |-> 0]
                ELSE IF bm = "caller" /\ c[t].n > 0 THEN [c[t] EXCEPT !.b = Sat(@ + 1)]
                ELSE c[t]
      new(t) == IF t # C0 /\ t \in rd[i]
                THEN [old(t) EXCEPT !.n = Sat(@ + 1), !.a = IF AlphaHere(s) THEN Sat(@ + 1) ELSE @]
                ELSE old(t)
  IN
  /\ l.kind = "gemm" /\ todo[i] = {<<"e", 0>>}
  /\ c' = [t \in Terms |-> new(t)]
  /\ cInit' = FALSE
  /\ bad' = bad \cup (IF Beta0 /\ bm # "zero" /\ cInit THEN {"read_c_when_beta0"} ELSE {})
                \cup (IF CScatter /\ ~vec THEN {"store_reads_junk_vec"} ELSE {})
  /\ todo' = [todo EXCEPT ![i] = {}]
  /\ UNCHANGED <<vec, buf, rd>>

Done == \A i \in LIdx : Finished(i)

Next ==
  \/ \E i \in LIdx : MayRun(i) /\
       \/ K0Step(i)
       \/ \E h \in 0..(K + L) : PackStep(i, h) \/ ReadStep(i, h)
       \/ Epi(i)
  \/ Done /\ UNCHANGED vars

Spec == Init /\ [][Next]_vars

\* ---------------------------------------------------------- properties --
NoBadStep == bad = {}

\* every position read exactly by the GEMM of its own slab
SlabsPartition ==
    /\ \A s \in Slabs : SlabPos(s) \subseteq 0..(K - 1)
    /\ \A h \in 0..(K - 1) : \E s \in Slabs : h \in SlabPos(s)
    /\ \A s1, s2 \in Slabs : s1 # s2 => SlabPos(s1) \cap SlabPos(s2) = {}

ResultExact ==
    Done =>
      /\ IF Beta0 THEN c[C0].n = 0
         ELSE c[C0] = [n |-> 1, a |-> 0, b |-> 1]
      /\ \A h \in 0..(K - 1) : c[h] = [n |-> 1, a |-> 1, b |-> 0]

TypeOK ==
  /\ vec \in BOOLEAN /\ cInit \in BOOLEAN
  /\ \A x \in Ops : DOMAIN buf[x] = 0..(Lhat - 1)

\* witness: a complete run exists (models are not vacuous)
NotDone == ~Done
=============================================================================
