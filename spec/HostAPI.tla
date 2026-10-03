------------------------------- MODULE HostAPI -------------------------------
(***************************************************************************)
(* Host-side concurrency and plan lifetime of the C API (DESIGN v1.2 §6).  *)
(*                                                                         *)
(* NT host threads each make one imd_contract call. A thread that owns a  *)
(* plan creates it first and destroys it last. Each call enqueues its GPU  *)
(* operations on its stream: K0 (writes the workspace's vec region; not in *)
(* the resident vector mode), a vector reader "K1" (stands for every       *)
(* kernel that reads the scatter vectors: K1-K3, K5) and the K4 GEMM       *)
(* launch (its kernel parameters). The GPU runs each stream in order,      *)
(* streams interleaved arbitrarily.                                        *)
(*                                                                         *)
(* What is checked (the v1.2 rules of §6):                                 *)
(*   NoHostRace      no two threads touch one host location concurrently   *)
(*                   with at least one write (a C++ data race). Every      *)
(*                   access to a shared location is a Begin/End pair, so   *)
(*                   an access of another thread in between is a race.     *)
(*   ParamsOwn       every GEMM is launched with the parameters built for  *)
(*                   its own call.                                         *)
(*   NoUseAfterFree  no GPU operation reads memory of a destroyed plan.    *)
(*   NoWsHazard      the GPU operations of two calls never interleave on   *)
(*                   one workspace.                                        *)
(*                                                                         *)
(* Design alternatives (constants):                                        *)
(*   VecSrc  "generate"    v1.2 IMD_VEC_GENERATE (default): K0 is a kernel *)
(*                         that generates the scatter vectors into the     *)
(*                         workspace from the fused-mode descriptor,       *)
(*                         passed BY VALUE as its kernel argument: no plan *)
(*                         memory is read on the GPU timeline;             *)
(*           "resident"    v1.2 IMD_VEC_RESIDENT: the plan owns a device   *)
(*                         copy of the vectors (built at plan creation);   *)
(*                         no K0, the kernels read the plan's copy;        *)
(*           "pinned"      v1.1: K0 is a cudaMemcpyAsync from the plan's   *)
(*                         pinned host buffer (read when the copy runs).   *)
(*   OpSrc   "planCache"   v1.2: the K4 GEMM parameters are built at plan  *)
(*                         creation (device properties, occupancy, smem    *)
(*                         attribute: all CUTLASS host state) and stored   *)
(*                         in the plan; imd_contract copies them to a      *)
(*                         local, patches the pointers and launches (the   *)
(*                         kernel parameters are copied at launch);        *)
(*           "perCall"     v1.1: a local CUTLASS operator per launch,      *)
(*                         initialized in imd_contract (with GlobalLazy =  *)
(*                         FALSE: CUTLASS v4.8.0 keeps its device          *)
(*                         properties in thread_local statics);            *)
(*           "lazyCache"   the plan's parameter cache is filled by the     *)
(*                         first imd_contract that finds it empty;         *)
(*           "sharedOp"    one mutable operator object per plan: a call    *)
(*                         writes its arguments into it, then launches     *)
(*                         from it.                                        *)
(*   GlobalLazy  imd_contract lazily initializes a process-global cache    *)
(*               (the unsynchronized device-property statics NOTES-M1 gap *)
(*               10 feared).                                               *)
(*   DestroyWaitsGPU  the plan owner synchronizes the streams of every     *)
(*               call using the plan before imd_plan_destroy (the v1.1     *)
(*               rule, and the v1.2 rule for resident plans); FALSE = the  *)
(*               v1.2 rule for generate plans: destroy once every call     *)
(*               using the plan has RETURNED on the host.                  *)
(*   SharePlan / ShareWs / SameStream: all threads use plan 1 / workspace  *)
(*               1 / stream 1 (else thread t uses plan / workspace /       *)
(*               stream t). A shared plan is created and destroyed by      *)
(*               thread 1; the others start using it once its creation    *)
(*               returned (the handle is published to them: happens-       *)
(*               before) and thread 1 destroys it after every user's call *)
(*               returned (the caller's ordering, as for any handle).      *)
(*                                                                         *)
(* Plan creation itself touches only the new plan (not yet visible to      *)
(* other threads) and thread-safe CUDA runtime calls (a resident plan      *)
(* builds and synchronizes its vectors before it returns), so it is one    *)
(* atomic step here; the CUDA runtime and driver are thread-safe and not   *)
(* modelled. Enqueueing is one step per operation: two threads launching  *)
(* on one stream interleave their operations.                             *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    NT,               \* host threads (one imd_contract call each)
    SharePlan,        \* every thread uses plan 1
    ShareWs,          \* every thread uses workspace 1
    SameStream,       \* every thread enqueues on stream 1
    VecSrc,           \* "generate" | "resident" | "pinned"
    OpSrc,            \* "planCache" | "perCall" | "lazyCache" | "sharedOp"
    GlobalLazy,       \* imd_contract lazily initializes a process-global cache
    DestroyWaitsGPU   \* destroy waits for the GPU work of the plan's calls

Th         == 1..NT
PlanOf(t)  == IF SharePlan THEN 1 ELSE t
WsOf(t)    == IF ShareWs THEN 1 ELSE t
StOf(t)    == IF SameStream THEN 1 ELSE t
Plans      == {PlanOf(t) : t \in Th}
Wss        == {WsOf(t) : t \in Th}
Streams    == {StOf(t) : t \in Th}
Users(p)   == {t \in Th : PlanOf(t) = p}
Owner(t)   == PlanOf(t) = t                   \* t creates and destroys its plan

\* The GPU operations of one call, in stream order (the last one ends the
\* call's use of its workspace).
Ops   == (IF VecSrc = "resident" THEN <<>> ELSE <<"K0">>) \o <<"K1", "gemm">>
NOps  == Len(Ops)

\* Shared host locations: the plan's parameter cache and operator object,
\* and one process-global cache.
Locs == {<<"cache", p>> : p \in Plans} \cup {<<"op", p>> : p \in Plans} \cup {<<"props", 0>>}
CacheLoc(p) == <<"cache", p>>
OpLoc(p)    == <<"op", p>>
PropsLoc    == <<"props", 0>>

\* The host program of thread t: plan creation (owner), the imd_contract
\* steps, then plan destruction (owner).
CallSteps ==
    (IF GlobalLazy THEN <<"propsB", "propsE">> ELSE <<>>)
    \o (CASE OpSrc = "planCache" -> <<"cacheRB", "cacheRE">>
          [] OpSrc = "lazyCache" -> <<"lazyB", "lazyE">>
          [] OpSrc = "perCall"   -> <<>>
          [] OpSrc = "sharedOp"  -> <<>>)
    \o (IF VecSrc = "resident" THEN <<>> ELSE <<"encK0">>)
    \o <<"encK1">>
    \o (IF OpSrc = "sharedOp" THEN <<"opWB", "opWE", "opRB", "opRE">> ELSE <<>>)
    \o <<"encGemm", "ret">>
Prog(t) == (IF Owner(t) THEN <<"create">> ELSE <<>>) \o <<"wait">> \o CallSteps
           \o (IF Owner(t) THEN <<"destroy">> ELSE <<>>)
RetIdx(t) == CHOOSE i \in DOMAIN Prog(t) : Prog(t)[i] = "ret"

ASSUME /\ NT \in Nat \ {0}
       /\ {SharePlan, ShareWs, SameStream, GlobalLazy, DestroyWaitsGPU} \subseteq BOOLEAN
       /\ VecSrc \in {"generate", "resident", "pinned"}
       /\ OpSrc \in {"planCache", "perCall", "lazyCache", "sharedOp"}

VARIABLES
    pc,        \* pc[t]: index of thread t's next host step in Prog(t)
    planSt,    \* planSt[p] \in {"none", "live", "dead"}
    cacheSt,   \* cacheSt[p] \in {"empty", "full"}: the plan's parameter cache
    propsSt,   \* the process-global lazy cache: "empty" | "full"
    opVal,     \* opVal[p]: the call whose arguments the shared operator holds
    par,       \* par[t]: the arguments thread t read back from the shared operator
    inacc,     \* inacc[loc]: accesses in progress, <<thread, "r" | "w">>
    queue,     \* queue[s]: GPU operations enqueued on stream s, not yet run
    wsOwn,     \* wsOwn[w]: the call whose GPU work is in progress on w (0 none)
    gpuDone,   \* gpuDone[t]: GPU operations of thread t's call that ran
    err        \* violations observed

vars == <<pc, planSt, cacheSt, propsSt, opVal, par, inacc, queue, wsOwn, gpuDone, err>>

Init ==
    /\ pc = [t \in Th |-> 1]
    /\ planSt = [p \in Plans |-> "none"]
    /\ cacheSt = [p \in Plans |-> "empty"]
    /\ propsSt = "empty"
    /\ opVal = [p \in Plans |-> 0]
    /\ par = [t \in Th |-> 0]
    /\ inacc = [l \in Locs |-> {}]
    /\ queue = [s \in Streams |-> <<>>]
    /\ wsOwn = [w \in Wss |-> 0]
    /\ gpuDone = [t \in Th |-> 0]
    /\ err = {}

-----------------------------------------------------------------------------
(* Host accesses *)

Conflict(t, l, kd) == \E x \in inacc[l] : x[1] # t /\ (kd = "w" \/ x[2] = "w")
\* Begin an access; a conflicting access of another thread in progress is a race.
Begin(t, l, kd) ==
    /\ err' = err \cup (IF Conflict(t, l, kd) THEN {"race"} ELSE {})
    /\ inacc' = [inacc EXCEPT ![l] = @ \cup {<<t, kd>>}]
Wrote(t, l) == <<t, "w">> \in inacc[l]
End(t, l)   == inacc' = [inacc EXCEPT ![l] = {x \in @ : x[1] # t}]

\* Plan memory each GPU operation of thread t's call reads (0: none).
PlanRef(t, kind) ==
    CASE kind = "K0" /\ VecSrc = "pinned"   -> PlanOf(t)
      [] kind = "K1" /\ VecSrc = "resident" -> PlanOf(t)
      [] OTHER                              -> 0
ParOf(t)  == IF OpSrc = "sharedOp" THEN par[t] ELSE t  \* whose arguments the GEMM gets
Enq(t, kind) ==
    /\ queue' = [queue EXCEPT ![StOf(t)] =
                   Append(@, [t |-> t, kind |-> kind, pref |-> PlanRef(t, kind),
                              par |-> IF kind = "gemm" THEN ParOf(t) ELSE t])]
    /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, inacc, wsOwn, gpuDone, err>>

HostStep(t) ==
    LET p == PlanOf(t)  st == Prog(t)[pc[t]] IN
    /\ pc[t] <= Len(Prog(t))
    /\ pc' = [pc EXCEPT ![t] = @ + 1]
    /\ CASE st = "create" ->
              \* builds the plan, including (planCache) the GEMM parameters
              /\ planSt' = [planSt EXCEPT ![p] = "live"]
              /\ cacheSt' = IF OpSrc = "planCache" THEN [cacheSt EXCEPT ![p] = "full"] ELSE cacheSt
              /\ UNCHANGED <<propsSt, opVal, par, inacc, queue, wsOwn, gpuDone, err>>
         [] st = "wait" ->      \* the plan handle is published to t
              /\ planSt[p] = "live"
              /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, inacc, queue, wsOwn, gpuDone, err>>
         [] st = "propsB" ->
              /\ Begin(t, PropsLoc, IF propsSt = "empty" THEN "w" ELSE "r")
              /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, queue, wsOwn, gpuDone>>
         [] st = "propsE" ->
              /\ propsSt' = IF Wrote(t, PropsLoc) THEN "full" ELSE propsSt
              /\ End(t, PropsLoc)
              /\ UNCHANGED <<planSt, cacheSt, opVal, par, queue, wsOwn, gpuDone, err>>
         [] st = "cacheRB" ->   \* copy the plan's parameters to a local
              /\ Begin(t, CacheLoc(p), "r")
              /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, queue, wsOwn, gpuDone>>
         [] st = "cacheRE" ->
              /\ End(t, CacheLoc(p))
              /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, queue, wsOwn, gpuDone, err>>
         [] st = "lazyB" ->
              /\ Begin(t, CacheLoc(p), IF cacheSt[p] = "empty" THEN "w" ELSE "r")
              /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, queue, wsOwn, gpuDone>>
         [] st = "lazyE" ->
              /\ cacheSt' = IF Wrote(t, CacheLoc(p)) THEN [cacheSt EXCEPT ![p] = "full"] ELSE cacheSt
              /\ End(t, CacheLoc(p))
              /\ UNCHANGED <<planSt, propsSt, opVal, par, queue, wsOwn, gpuDone, err>>
         [] st = "opWB" ->
              /\ Begin(t, OpLoc(p), "w")
              /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, queue, wsOwn, gpuDone>>
         [] st = "opWE" ->
              /\ opVal' = [opVal EXCEPT ![p] = t]
              /\ End(t, OpLoc(p))
              /\ UNCHANGED <<planSt, cacheSt, propsSt, par, queue, wsOwn, gpuDone, err>>
         [] st = "opRB" ->
              /\ Begin(t, OpLoc(p), "r")
              /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, queue, wsOwn, gpuDone>>
         [] st = "opRE" ->
              /\ par' = [par EXCEPT ![t] = opVal[p]]
              /\ End(t, OpLoc(p))
              /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, queue, wsOwn, gpuDone, err>>
         [] st = "encK0"   -> Enq(t, "K0")
         [] st = "encK1"   -> Enq(t, "K1")
         [] st = "encGemm" -> Enq(t, "gemm")
         [] st = "ret" ->       \* imd_contract returns
              UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, inacc, queue, wsOwn, gpuDone, err>>
         [] st = "destroy" ->
              /\ \A u \in Users(p) : pc[u] > RetIdx(u)
              /\ DestroyWaitsGPU => \A u \in Users(p) : gpuDone[u] = NOps
              /\ planSt' = [planSt EXCEPT ![p] = "dead"]
              /\ UNCHANGED <<cacheSt, propsSt, opVal, par, inacc, queue, wsOwn, gpuDone, err>>

\* The GPU runs the head operation of stream s.
GpuStep(s) ==
    LET op == Head(queue[s])  w == WsOf(op.t) IN
    /\ queue[s] # <<>>
    /\ queue' = [queue EXCEPT ![s] = Tail(@)]
    /\ err' = err
              \cup (IF op.pref # 0 /\ planSt[op.pref] = "dead" THEN {"uaf"} ELSE {})
              \cup (IF wsOwn[w] \notin {0, op.t} THEN {"wsHazard"} ELSE {})
              \cup (IF op.kind = "gemm" /\ op.par # op.t THEN {"wrongParams"} ELSE {})
    /\ wsOwn' = [wsOwn EXCEPT ![w] = IF op.kind = "gemm" THEN 0 ELSE op.t]
    /\ gpuDone' = [gpuDone EXCEPT ![op.t] = @ + 1]
    /\ UNCHANGED <<pc, planSt, cacheSt, propsSt, opVal, par, inacc>>

AllDone == /\ \A t \in Th : pc[t] > Len(Prog(t))
           /\ \A s \in Streams : queue[s] = <<>>

Next == \/ \E t \in Th : HostStep(t)
        \/ \E s \in Streams : GpuStep(s)
        \/ (AllDone /\ UNCHANGED vars)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

-----------------------------------------------------------------------------
(* Invariants and properties *)

TypeOK ==
    /\ pc \in [Th -> Nat]
    /\ planSt \in [Plans -> {"none", "live", "dead"}]
    /\ cacheSt \in [Plans -> {"empty", "full"}]
    /\ propsSt \in {"empty", "full"}
    /\ opVal \in [Plans -> 0..NT] /\ par \in [Th -> 0..NT]
    /\ gpuDone \in [Th -> 0..NOps]
    /\ err \subseteq {"race", "uaf", "wsHazard", "wrongParams"}

NoHostRace     == "race" \notin err
NoUseAfterFree == "uaf" \notin err
NoWsHazard     == "wsHazard" \notin err
ParamsOwn      == "wrongParams" \notin err
\* Every call's GPU work ran and every plan was destroyed.
CompleteAtEnd  == AllDone => /\ \A t \in Th : gpuDone[t] = NOps
                             /\ \A p \in Plans : planSt[p] = "dead"
Termination    == <>AllDone

\* Witnesses (expected to be VIOLATED: they show a positive model reaches the
\* situation its rule is about, so it does not pass vacuously).
\*   a plan is destroyed while GPU work of a call that used it is pending:
NeverDestroyedEarly ==
    ~\E p \in Plans : planSt[p] = "dead" /\ \E u \in Users(p) : gpuDone[u] < NOps
\*   two threads read one plan location at the same time:
NeverConcurrentRead == \A l \in Locs : Cardinality({x[1] : x \in inacc[l]}) <= 1
\*   the GPU operations of two calls interleave on one stream:
NeverInterleaved ==
    \A s \in Streams : ~\E i, q, j \in DOMAIN queue[s] :
        /\ i < q /\ q < j
        /\ queue[s][i].t = queue[s][j].t /\ queue[s][q].t # queue[s][i].t
=============================================================================
