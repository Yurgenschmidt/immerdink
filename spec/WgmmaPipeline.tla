---------------------------- MODULE WgmmaPipeline ----------------------------
(***************************************************************************)
(* The load / compute hand-off of a Hopper K4 (issue 17, docs/NOTES-17.md). *)
(*                                                                         *)
(* A persistent CTA runs one producer (a thread that issues TMA copies)    *)
(* and NC consumer warpgroups (wgmma).  Between them: S shared-memory       *)
(* stages, and per stage two mbarriers.  FULL says "the stage holds its     *)
(* data"; EMPTY says "every reader is done with it".  The model is the     *)
(* protocol of CUTLASS's PipelineTmaAsync / sm90 warp-specialized kernels:  *)
(*                                                                         *)
(*  producer, iteration i (stage i mod S, round i div S):                  *)
(*      wait EMPTY[stage] (parity 1 - round mod 2: the first round passes) *)
(*      arrive on FULL[stage] with the expected bytes; start the TMA copies *)
(*      (they land later, asynchronously, and complete FULL's byte count)   *)
(*  consumer, iteration i:                                                 *)
(*      wait FULL[stage] (parity round mod 2); issue wgmma (an ASYNC read   *)
(*      of the stage); keep at most Depth wgmma in flight: retire the       *)
(*      oldest, then arrive on EMPTY of its stage; at the end of a tile    *)
(*      retire everything, run the epilogue (registers only: K4's          *)
(*      epilogue uses no shared memory), and go on to the next tile.        *)
(*                                                                         *)
(* Mode "coop": all NC warpgroups work on every tile (each on its rows).   *)
(* Mode "ping": warpgroup w works on tiles t with t mod NC = w - 1; every  *)
(*      consumer still steps its pipeline position past the iterations of   *)
(*      the other warpgroup's tiles (CUTLASS advances the state).          *)
(* Cl = 2: a cluster of two CTAs; each CTA's producer loads one half of    *)
(*      the stage into BOTH CTAs (TMA multicast).  FULL of a CTA completes  *)
(*      when its own arrival and both halves have landed; EMPTY of a CTA     *)
(*      needs the arrivals of the consumers of both CTAs.                   *)
(*                                                                         *)
(* Memory is sequentially consistent and time is ignored (as the other     *)
(* specs here); the point is the ORDER of the hand-off.                    *)
(*                                                                         *)
(* Checked: NoOverwrite (a copy never lands in a stage a consumer holds or  *)
(* an async wgmma still reads), ReadsCorrect (a wgmma reads the stage's    *)
(* data of exactly the iteration it expects, all halves landed),           *)
(* AccExact (at the epilogue a warpgroup has accumulated k = 0..KT-1 of its *)
(* tile once each, in order), and no deadlock before every tile is done    *)
(* (TLC's deadlock check; the finished state stutters).  Mut names one      *)
(* realistic bug; every mutant has a model that fails.                     *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, Sequences

CONSTANTS S,      \* stages
          KT,     \* k-steps per tile
          T,      \* tiles in the CTA's work list
          NC,     \* consumer warpgroups per CTA
          Cl,     \* CTAs in the cluster (1 or 2)
          Mode,   \* "coop" | "ping"
          Depth,  \* wgmma allowed in flight after an issue (0 or 1)
          Mut     \* "none" or the name of a mutant

ASSUME S \in 1..8 /\ KT \in 1..4 /\ T \in 1..4 /\ NC \in 1..2 /\ Cl \in 1..2
ASSUME Mode \in {"coop", "ping"} /\ Depth \in 0..1
ASSUME Mode = "ping" => NC = 2

CTAs   == 1..Cl
WGs    == 1..NC
Stages == 0..(S - 1)
N      == T * KT                      \* iterations of the producer
Tiles  == 0..(T - 1)

TileOf(i)  == i \div KT
KOf(i)     == i % KT
StageOf(i) == i % S
RoundOf(i) == i \div S

OwnTiles(w) == IF Mode = "coop" THEN Tiles ELSE {t \in Tiles : (t % NC) = w - 1}
OwnCount(w) == Cardinality(OwnTiles(w)) * KT
\* the n-th iteration (0-based) of consumer w, in order
IterOf(w, n) == IF Mode = "coop" THEN n
                ELSE ((w - 1) + NC * (n \div KT)) * KT + (n % KT)

\* ------------------------------------------------------------------ mutants --
\* consumers that read one stage: both warpgroups (coop) or the tile's owner (ping)
PerStage == IF Mode = "coop" THEN NC ELSE 1
ExpEmpty == IF Mut = "empty_count_low" THEN PerStage * Cl - 1
            ELSE IF Mut = "empty_local_only" THEN PerStage
            ELSE PerStage * Cl
ProdParity(i) == IF Mut = "producer_parity_wrong" THEN RoundOf(i) % 2
                 ELSE ((RoundOf(i) + 1) % 2)
ConsParity(i) == IF Mut = "consumer_parity_wrong" THEN ((RoundOf(i) + 1) % 2)
                 ELSE RoundOf(i) % 2

VARIABLES
  smem,      \* [CTAs \X Stages -> [tag: Int, parts: SUBSET CTAs]]  what the stage holds
  full,      \* [CTAs \X Stages -> [ph: 0..1, arr: BOOLEAN, it: Int]] FULL mbarrier
  empty,     \* [CTAs \X Stages -> [ph: 0..1, cnt: Nat]]               EMPTY mbarrier
  pi,        \* [CTAs -> 0..N]  next iteration the producer issues
  tma,       \* set of copies in flight: [src, dst, st, it]
  cn,        \* [CTAs \X WGs -> consumer state]
  held,      \* [CTAs \X Stages -> SUBSET WGs]  acquired and not yet released
  rd,        \* [CTAs \X Stages -> SUBSET WGs]  an async wgmma may still read
  bad        \* set of violation names

vars == <<smem, full, empty, pi, tma, cn, held, rd, bad>>

Idx == CTAs \X Stages
CIdx == CTAs \X WGs

\* n: own iterations issued, r: own iterations retired, acc: the k values accumulated into the
\* current tile in order, done: tiles whose epilogue ran
ConsInit == [n |-> 0, r |-> 0, acc |-> << >>, done |-> 0]

Init ==
  /\ smem = [x \in Idx |-> [tag |-> -1, parts |-> {}]]
  /\ full = [x \in Idx |-> [ph |-> 0, arr |-> FALSE, it |-> -1]]
  /\ empty = [x \in Idx |-> [ph |-> 0, cnt |-> 0]]
  /\ pi = [c \in CTAs |-> 0]
  /\ tma = {}
  /\ cn = [x \in CIdx |-> ConsInit]
  /\ held = [x \in Idx |-> {}]
  /\ rd = [x \in Idx |-> {}]
  /\ bad = {}

\* ----------------------------------------------------------------- producer --
Produce(c) ==
  LET i == pi[c] IN
  /\ i < N
  /\ \/ Mut = "no_empty_wait"
     \/ empty[<<c, StageOf(i)>>].ph # ProdParity(i)
  /\ full' = [full EXCEPT ![<<c, StageOf(i)>>] = [ph |-> @.ph, arr |-> TRUE, it |-> i]]
  /\ tma' = tma \cup {[src |-> c, dst |-> d, st |-> StageOf(i), it |-> i] : d \in CTAs}
  /\ pi' = [pi EXCEPT ![c] = @ + 1]
  /\ UNCHANGED <<smem, empty, cn, held, rd, bad>>

\* a copy lands: the half of the stage that CTA src loads, in CTA dst's shared memory
Land(x) ==
  /\ x \in tma
  /\ LET k == <<x.dst, x.st>>
         old == smem[k]
         fresh == old.tag # x.it
         clash == held[k] # {} \/ rd[k] # {}
     IN /\ smem' = [smem EXCEPT ![k] = [tag |-> x.it,
                                         parts |-> IF fresh THEN {x.src} ELSE old.parts \cup {x.src}]]
        /\ bad' = IF clash THEN bad \cup {"overwrite"} ELSE bad
  /\ tma' = tma \ {x}
  /\ UNCHANGED <<full, empty, pi, cn, held, rd>>

\* FULL completes: the producer arrived (with the expected byte count) and every half landed
FullDone(k) ==
  /\ full[k].arr
  /\ smem[k].tag = full[k].it
  /\ smem[k].parts = CTAs
  /\ full' = [full EXCEPT ![k] = [ph |-> 1 - @.ph, arr |-> FALSE, it |-> @.it]]
  /\ UNCHANGED <<smem, empty, pi, tma, cn, held, rd, bad>>

\* ----------------------------------------------------------------- consumers --
GId(w, n) == IterOf(w, n)                \* global iteration of own iteration n
PosOf(w, n) == IF Mut = "no_advance_pingpong" THEN n ELSE GId(w, n)   \* the pipeline position it waits at

Arrive(e) ==   \* one arrival on an EMPTY barrier
  [ph |-> IF e.cnt + 1 = ExpEmpty THEN 1 - e.ph ELSE e.ph,
   cnt |-> IF e.cnt + 1 = ExpEmpty THEN 0 ELSE e.cnt + 1]

\* the arrivals of a consumer of CTA c that is done with pipeline stage st
ReleaseEff(c, st) ==
  LET tgt == IF Mut = "empty_local_only" THEN {c} ELSE CTAs
      once == [x \in Idx |-> IF x[1] \in tgt /\ x[2] = st THEN Arrive(empty[x]) ELSE empty[x]]
  IN IF Mut = "double_release"
     THEN [x \in Idx |-> IF x[1] \in tgt /\ x[2] = st THEN Arrive(once[x]) ELSE once[x]]
     ELSE once

TileBoundary(s) == s.n > 0 /\ s.n % KT = 0       \* every iteration of a tile issued
EpiDue(s) == TileBoundary(s) /\ s.r = s.n /\ s.done < s.n \div KT

\* Pingpong's order barrier: warpgroup w starts the mainloop of tile t only after the other warpgroup
\* has issued every iteration of tile t - 1.  It is NOT only a utilization device: a parity wait passes
\* when the barrier's phase bit differs from the parity, so a warpgroup that waits for round r of a
\* stage before round r - 1 completed passes at once (the bit is still that of round r - 2).  Because the
\* other warpgroup's waits all passed, every earlier round of every stage is complete.
OrderOK(c, w, i) ==
  \/ Mode = "coop"
  \/ Mut = "no_order_barrier"
  \/ TileOf(i) = 0
  \/ \E o \in WGs \ {w} : cn[<<c, o>>].n >= (((TileOf(i) - 1) \div NC) + 1) * KT

Issue(c, w) ==
  LET s == cn[<<c, w>>]
      n == s.n
      i == GId(w, n)
      pos == PosOf(w, n)
      k == <<c, pos % S>>
      okdata == smem[k].tag = i /\ smem[k].parts = CTAs
      early == Mut = "release_early"
  IN
  /\ n < OwnCount(w)
  /\ ~(TileBoundary(s) /\ s.done < n \div KT)     \* the previous tile's epilogue first
  /\ OrderOK(c, w, i)
  /\ \/ Mut = "no_full_wait"
     \/ full[k].ph # ConsParity(pos)
  /\ bad' = IF okdata THEN bad ELSE bad \cup {"data"}
  /\ held' = IF early THEN held ELSE [held EXCEPT ![k] = @ \cup {w}]
  /\ rd' = [rd EXCEPT ![k] = @ \cup {w}]
  /\ empty' = IF early THEN ReleaseEff(c, pos % S) ELSE empty
  /\ cn' = [cn EXCEPT ![<<c, w>>] = [s EXCEPT !.n = n + 1, !.acc = Append(@, KOf(i))]]
  /\ UNCHANGED <<smem, full, pi, tma>>

\* retire the oldest wgmma (while more than Depth are in flight, or when the tile is fully issued),
\* then release its stage
Retire(c, w) ==
  LET s == cn[<<c, w>>]
      i == GId(w, s.r)
      pos == PosOf(w, s.r)
      st == pos % S
      k == <<c, st>>
      early == Mut = "release_early"
      skip == Mut = "no_release_last" /\ KOf(i) = KT - 1
  IN
  /\ s.r < s.n
  /\ \/ s.n - s.r > Depth
     \/ TileBoundary(s)
  /\ rd' = [rd EXCEPT ![k] = @ \ {w}]
  /\ held' = IF early \/ skip THEN held ELSE [held EXCEPT ![k] = @ \ {w}]
  /\ empty' = IF early \/ skip THEN empty ELSE ReleaseEff(c, st)
  /\ cn' = [cn EXCEPT ![<<c, w>>] = [s EXCEPT !.r = @ + 1]]
  /\ UNCHANGED <<smem, full, pi, tma, bad>>

\* the epilogue (registers only): the tile's accumulation must be exactly k = 0 .. KT-1 in order
Epilogue(c, w) ==
  LET s == cn[<<c, w>>] IN
  /\ EpiDue(s)
  /\ bad' = IF s.acc = [j \in 1..KT |-> j - 1] THEN bad ELSE bad \cup {"acc"}
  /\ cn' = [cn EXCEPT ![<<c, w>>] =
              [s EXCEPT !.done = @ + 1, !.acc = IF Mut = "acc_not_reset" THEN @ ELSE << >>]]
  /\ UNCHANGED <<smem, full, empty, pi, tma, held, rd>>

AllDone ==
  /\ \A c \in CTAs : pi[c] = N
  /\ tma = {}
  /\ \A c \in CTAs, w \in WGs : cn[<<c, w>>].done = Cardinality(OwnTiles(w))

Next ==
  \/ \E c \in CTAs : Produce(c)
  \/ \E x \in tma : Land(x)
  \/ \E k \in Idx : FullDone(k)
  \/ \E c \in CTAs, w \in WGs : Issue(c, w) \/ Retire(c, w) \/ Epilogue(c, w)
  \/ AllDone /\ UNCHANGED vars

Spec == Init /\ [][Next]_vars

\* ---------------------------------------------------------------- properties --
TypeOK ==
  /\ \A x \in Idx : smem[x].parts \subseteq CTAs /\ full[x].ph \in 0..1 /\ empty[x].ph \in 0..1
  /\ \A x \in CIdx : cn[x].r <= cn[x].n /\ cn[x].n - cn[x].r <= S

NoOverwrite == "overwrite" \notin bad
ReadsCorrect == "data" \notin bad
AccExact == "acc" \notin bad

\* Witnesses (negative models): the situations the positive models must reach are reachable.
AllStagesArmed == ~(\E c \in CTAs : \A st \in Stages : full[<<c, st>>].arr)   \* the pipeline fills
TwoInFlight == ~(\E x \in CIdx : cn[x].n - cn[x].r = 2)                         \* Depth 1 overlaps wgmma
PeerFirst == ~(\E k \in Idx : smem[k].parts # {} /\ smem[k].tag # full[k].it)   \* a peer's half lands first
BothConsume == ~(\E k \in Idx : Cardinality(rd[k]) = 2)                         \* two warpgroups read one stage
=============================================================================
