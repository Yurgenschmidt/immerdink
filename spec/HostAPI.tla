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
(* D1 pack handle (DESIGN §6b.7; PackMode # "none"): thread 1 creates the  *)
(* imd_pack_t H1 (an immutable host object: the key and the layout) and   *)
(* publishes it; every thread then passes it to imd_pack (a host read of  *)
(* the handle, then the pack kernel: two GPU steps pk1/pk2 that own the    *)
(* thread's packed buffer while they run), every plan owner to            *)
(* imd_plan_create_packed (a host read; the plan COPIES the key), and     *)
(* every imd_contract_packed compares the key of the handle it is given   *)
(* with the plan's copy (a host read). Thread 1 destroys H1 once every    *)
(* call that was passed it has RETURNED, GPU work possibly pending (no GPU *)
(* work reads the handle: imd_pack's kernel arguments are copied at       *)
(* launch). NewHandle: thread 1 also creates H2 with the same key and the *)
(* contracts are given H2, so H1 is destroyed right after the imd_pack    *)
(* calls returned and every plan OUTLIVES the handle it was created with. *)
(*   PackMode "design"     as above;                                       *)
(*            "lazyCache"  the handle has a cache filled by the first call *)
(*                         that finds it empty (a write);                  *)
(*            "planPtr"    the plan keeps a pointer to the handle instead  *)
(*                         of copying the key: the contract reads H1      *)
(*                         through it;                                     *)
(*            "gpuReads"   the pack kernel reads the handle (e.g. a        *)
(*                         pointer to the layout instead of a by-value     *)
(*                         kernel argument).                               *)
(*   ShareBuf  every thread's imd_pack writes packed buffer 1 (two imd_pack *)
(*             calls into one buffer, like a workspace shared by calls).   *)
(*   NoPackBufHazard  the GPU work of two imd_pack calls never interleaves *)
(*             on one packed buffer.                                       *)
(* (The packed buffer's RAW / WAR ordering against the contracts that read *)
(* it is the caller's: PackedOperand StreamSpec.)                          *)
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
    DestroyWaitsGPU,  \* destroy waits for the GPU work of the plan's calls
    PackMode,         \* "none" (no pack handle) | "design" | "lazyCache" |
                      \*   "planPtr" | "gpuReads"  (D1, DESIGN §6b.7)
    NewHandle,        \* the contracts are given a second handle H2 (same key)
    ShareBuf          \* every imd_pack writes packed buffer 1

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

\* D1 pack handles: H1 (and with NewHandle H2), both created by thread 1.
UsePack   == PackMode # "none"
Hs        == IF ~UsePack THEN {} ELSE IF NewHandle THEN {1, 2} ELSE {1}
ContractH == IF NewHandle THEN 2 ELSE 1         \* the handle the contracts are given
PbOf(t)   == IF ShareBuf THEN 1 ELSE t          \* the packed buffer t's imd_pack writes
Pbs       == IF UsePack THEN {PbOf(t) : t \in Th} ELSE {}
\* Every GPU operation of thread t: its imd_pack's kernel (two steps that
\* own the packed buffer while it runs), then its contract's.
NAll  == (IF UsePack THEN 2 ELSE 0) + NOps

\* Shared host locations: the plan's parameter cache and operator object,
\* one process-global cache, and the pack handles.
Locs == {<<"cache", p>> : p \in Plans} \cup {<<"op", p>> : p \in Plans} \cup {<<"props", 0>>}
        \cup {<<"pack", h>> : h \in Hs}
CacheLoc(p) == <<"cache", p>>
OpLoc(p)    == <<"op", p>>
PropsLoc    == <<"props", 0>>
HLoc(h)     == <<"pack", h>>

\* The host program of thread t: plan creation (owner), the imd_contract
\* steps, then plan destruction (owner).
CallSteps ==
    (IF GlobalLazy THEN <<"propsB", "propsE">> ELSE <<>>)
    \o (CASE OpSrc = "planCache" -> <<"cacheRB", "cacheRE">>
          [] OpSrc = "lazyCache" -> <<"lazyB", "lazyE">>
          [] OpSrc = "perCall"   -> <<>>
          [] OpSrc = "sharedOp"  -> <<>>)
    \o (IF UsePack THEN <<"hRBc", "hREc">> ELSE <<>>)      \* compare the given handle's key
    \o (IF PackMode = "planPtr" THEN <<"ptrRB", "ptrRE">> ELSE <<>>)  \* with the plan's
    \o (IF VecSrc = "resident" THEN <<>> ELSE <<"encK0">>)
    \o <<"encK1">>
    \o (IF OpSrc = "sharedOp" THEN <<"opWB", "opWE", "opRB", "opRE">> ELSE <<>>)
    \o <<"encGemm", "ret">>
\* With a pack handle: thread 1 creates H1 (and H2), the others wait for
\* them to be published; a plan owner reads H1 to create its plan
\* (imd_plan_create_packed); every thread calls imd_pack(H1) (host read,
\* two launches, return) before its contract; thread 1 destroys H1 after
\* the last call that was passed H1 returned (NewHandle: right after its
\* own imd_pack; H2 at the end).
PackHead(t) == IF ~UsePack THEN <<>>
               ELSE IF t = 1 THEN <<"hcreate1">> \o (IF NewHandle THEN <<"hcreate2">> ELSE <<>>)
               ELSE <<"waitH">>
PlanCreate(t) == IF ~Owner(t) THEN <<>>
                 ELSE (IF UsePack THEN <<"hRB1", "hRE1">> ELSE <<>>) \o <<"create">>
PackCall(t) == IF ~UsePack THEN <<>>
               ELSE <<"hRB1", "hRE1", "encPk1", "encPk2", "pret">>
                    \o (IF t = 1 /\ NewHandle THEN <<"hdestroy1">> ELSE <<>>)
PackTail(t) == IF ~UsePack \/ t # 1 THEN <<>>
               ELSE IF NewHandle THEN <<"hdestroy2">> ELSE <<"hdestroy1">>
Prog(t) == PackHead(t) \o PlanCreate(t) \o <<"wait">> \o PackCall(t) \o CallSteps
           \o (IF Owner(t) THEN <<"destroy">> ELSE <<>>) \o PackTail(t)
RetIdx(t)  == CHOOSE i \in DOMAIN Prog(t) : Prog(t)[i] = "ret"
PRetIdx(t) == CHOOSE i \in DOMAIN Prog(t) : Prog(t)[i] = "pret"
\* The last return of a call of thread t that was passed handle h (DESIGN
\* §6b.7: h may be destroyed once all of them have returned).
LastUse(t, h) == IF h = 1 /\ NewHandle THEN PRetIdx(t) ELSE RetIdx(t)

ASSUME /\ NT \in Nat \ {0}
       /\ {SharePlan, ShareWs, SameStream, GlobalLazy, DestroyWaitsGPU} \subseteq BOOLEAN
       /\ VecSrc \in {"generate", "resident", "pinned"}
       /\ OpSrc \in {"planCache", "perCall", "lazyCache", "sharedOp"}
       /\ PackMode \in {"none", "design", "lazyCache", "planPtr", "gpuReads"}
       /\ {NewHandle, ShareBuf} \subseteq BOOLEAN
       /\ ~UsePack => ~NewHandle /\ ~ShareBuf

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
    gpuDone,   \* gpuDone[t]: GPU operations of thread t's calls that ran
    err,       \* violations observed
    hSt,       \* hSt[h] \in {"none", "live", "dead"}: pack handle h
    hc,        \* hc[h] \in {"empty", "full"}: the handle's lazy cache ("lazyCache")
    pbOwn      \* pbOwn[b]: the call whose imd_pack runs on packed buffer b (0: none)

hvars == <<hSt, hc, pbOwn>>
vars  == <<pc, planSt, cacheSt, propsSt, opVal, par, inacc, queue, wsOwn, gpuDone, err, hvars>>

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
    /\ hSt = [h \in Hs |-> "none"]
    /\ hc = [h \in Hs |-> "empty"]
    /\ pbOwn = [x \in Pbs |-> 0]

-----------------------------------------------------------------------------
(* Host accesses *)

Conflict(t, l, kd) == \E x \in inacc[l] : x[1] # t /\ (kd = "w" \/ x[2] = "w")
\* Begin an access; a conflicting access of another thread in progress is a race.
Begin(t, l, kd) ==
    /\ err' = err \cup (IF Conflict(t, l, kd) THEN {"race"} ELSE {})
    /\ inacc' = [inacc EXCEPT ![l] = @ \cup {<<t, kd>>}]
Wrote(t, l) == <<t, "w">> \in inacc[l]
End(t, l)   == inacc' = [inacc EXCEPT ![l] = {x \in @ : x[1] # t}]
\* A host access to pack handle h: a read (the handle is immutable), or
\* with a lazy cache a write while the cache is empty; touching a handle
\* that is not live is a (host) use after free.
HKind(h) == IF PackMode = "lazyCache" /\ hc[h] = "empty" THEN "w" ELSE "r"
HBegin(t, h, kd) ==
    /\ err' = err \cup (IF Conflict(t, HLoc(h), kd) THEN {"race"} ELSE {})
                  \cup (IF hSt[h] # "live" THEN {"uaf"} ELSE {})
    /\ inacc' = [inacc EXCEPT ![HLoc(h)] = @ \cup {<<t, kd>>}]
HEnd(t, h) ==
    /\ hc' = IF Wrote(t, HLoc(h)) THEN [hc EXCEPT ![h] = "full"] ELSE hc
    /\ End(t, HLoc(h))

\* Plan memory each GPU operation of thread t's call reads (0: none).
PlanRef(t, kind) ==
    CASE kind = "K0" /\ VecSrc = "pinned"   -> PlanOf(t)
      [] kind = "K1" /\ VecSrc = "resident" -> PlanOf(t)
      [] OTHER                              -> 0
ParOf(t)  == IF OpSrc = "sharedOp" THEN par[t] ELSE t  \* whose arguments the GEMM gets
\* The pack handle a GPU operation reads (0: none). By design none: the
\* pack kernel's arguments are copied at launch.
HandleRef(kind) == IF PackMode = "gpuReads" /\ kind \in {"pk1", "pk2"} THEN 1 ELSE 0
Enq(t, kind) ==
    /\ queue' = [queue EXCEPT ![StOf(t)] =
                   Append(@, [t |-> t, kind |-> kind, pref |-> PlanRef(t, kind),
                              href |-> HandleRef(kind),
                              par |-> IF kind = "gemm" THEN ParOf(t) ELSE t])]
    /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, inacc, wsOwn, gpuDone, err>>

HandleSteps == {"hcreate1", "hcreate2", "waitH", "hRB1", "hRE1", "hRBc", "hREc",
                "ptrRB", "ptrRE", "hdestroy1", "hdestroy2"}
\* The host steps on pack handles (the GPU side is in GpuStep).
HandleStep(t, st) ==
    LET h == CASE st \in {"hcreate2", "hdestroy2"} -> 2
                [] st \in {"hRBc", "hREc"}          -> ContractH
                [] OTHER                            -> 1
    IN
    CASE st \in {"hcreate1", "hcreate2"} ->    \* imd_pack_create: a new host object
           /\ hSt' = [hSt EXCEPT ![h] = "live"]
           /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, inacc, queue, wsOwn, gpuDone,
                          err, hc, pbOwn>>
      [] st = "waitH" ->                       \* the handles are published to t
           /\ \A h2 \in Hs : hSt[h2] = "live"
           /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, inacc, queue, wsOwn, gpuDone,
                          err, hvars>>
      [] st \in {"hRB1", "hRBc"} ->             \* read the handle's key / layout
           /\ HBegin(t, h, HKind(h))
           /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, queue, wsOwn, gpuDone, hvars>>
      [] st = "ptrRB" ->                       \* planPtr: H1's key through the plan's pointer
           /\ HBegin(t, h, "r")
           /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, queue, wsOwn, gpuDone, hvars>>
      [] st \in {"hRE1", "hREc", "ptrRE"} ->
           /\ HEnd(t, h)
           /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, queue, wsOwn, gpuDone, err,
                          hSt, pbOwn>>
      [] st \in {"hdestroy1", "hdestroy2"} ->   \* imd_pack_destroy: every call passed h returned
           /\ \A u \in Th : pc[u] > LastUse(u, h)
           /\ hSt' = [hSt EXCEPT ![h] = "dead"]
           /\ UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, inacc, queue, wsOwn, gpuDone,
                          err, hc, pbOwn>>

HostStep(t) ==
    LET p == PlanOf(t)  st == Prog(t)[pc[t]] IN
    /\ pc[t] <= Len(Prog(t))
    /\ pc' = [pc EXCEPT ![t] = @ + 1]
    /\ st \in HandleSteps => HandleStep(t, st)
    /\ st \notin HandleSteps => UNCHANGED hvars
    /\ st \notin HandleSteps =>
       CASE st = "create" ->
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
         [] st = "encPk1"  -> Enq(t, "pk1")
         [] st = "encPk2"  -> Enq(t, "pk2")
         [] st \in {"ret", "pret"} ->       \* imd_contract / imd_pack returns
              UNCHANGED <<planSt, cacheSt, propsSt, opVal, par, inacc, queue, wsOwn, gpuDone, err>>
         [] st = "destroy" ->
              /\ \A u \in Users(p) : pc[u] > RetIdx(u)
              /\ DestroyWaitsGPU => \A u \in Users(p) : gpuDone[u] = NAll
              /\ planSt' = [planSt EXCEPT ![p] = "dead"]
              /\ UNCHANGED <<cacheSt, propsSt, opVal, par, inacc, queue, wsOwn, gpuDone, err>>

\* The GPU runs the head operation of stream s. A contract's operations own
\* its workspace from the first one to the GEMM; an imd_pack's two steps
\* own its packed buffer (pk1 starts writing it, pk2 finishes).
GpuStep(s) ==
    LET op == Head(queue[s])  w == WsOf(op.t)  isPk == op.kind \in {"pk1", "pk2"}
        pb == PbOf(op.t) IN
    /\ queue[s] # <<>>
    /\ queue' = [queue EXCEPT ![s] = Tail(@)]
    /\ err' = err
              \cup (IF op.pref # 0 /\ planSt[op.pref] = "dead" THEN {"uaf"} ELSE {})
              \cup (IF op.href # 0 /\ hSt[op.href] = "dead" THEN {"uaf"} ELSE {})
              \cup (IF ~isPk /\ wsOwn[w] \notin {0, op.t} THEN {"wsHazard"} ELSE {})
              \cup (IF isPk /\ pbOwn[pb] \notin {0, op.t} THEN {"pbHazard"} ELSE {})
              \cup (IF op.kind = "gemm" /\ op.par # op.t THEN {"wrongParams"} ELSE {})
    /\ wsOwn' = IF isPk THEN wsOwn
                ELSE [wsOwn EXCEPT ![w] = IF op.kind = "gemm" THEN 0 ELSE op.t]
    /\ pbOwn' = IF isPk THEN [pbOwn EXCEPT ![pb] = IF op.kind = "pk2" THEN 0 ELSE op.t]
                ELSE pbOwn
    /\ gpuDone' = [gpuDone EXCEPT ![op.t] = @ + 1]
    /\ UNCHANGED <<pc, planSt, cacheSt, propsSt, opVal, par, inacc, hSt, hc>>

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
    /\ gpuDone \in [Th -> 0..NAll]
    /\ err \subseteq {"race", "uaf", "wsHazard", "wrongParams", "pbHazard"}
    /\ hSt \in [Hs -> {"none", "live", "dead"}] /\ hc \in [Hs -> {"empty", "full"}]
    /\ pbOwn \in [Pbs -> 0..NT]

NoHostRace      == "race" \notin err
NoUseAfterFree  == "uaf" \notin err          \* plans and pack handles, GPU and host
NoWsHazard      == "wsHazard" \notin err
ParamsOwn       == "wrongParams" \notin err
NoPackBufHazard == "pbHazard" \notin err
\* Every call's GPU work ran and every plan and pack handle was destroyed.
CompleteAtEnd  == AllDone => /\ \A t \in Th : gpuDone[t] = NAll
                             /\ \A p \in Plans : planSt[p] = "dead"
                             /\ \A h \in Hs : hSt[h] = "dead"
Termination    == <>AllDone

\* Witnesses (expected to be VIOLATED: they show a positive model reaches the
\* situation its rule is about, so it does not pass vacuously).
\*   a plan is destroyed while GPU work of a call that used it is pending:
NeverDestroyedEarly ==
    ~\E p \in Plans : planSt[p] = "dead" /\ \E u \in Users(p) : gpuDone[u] < NAll
\*   a pack handle is destroyed while an imd_pack kernel it was passed to is
\*   still pending on the GPU (the first two GPU operations of each thread):
NeverHandleDestroyedEarly ==
    ~\E h \in Hs : hSt[h] = "dead" /\ \E t \in Th : gpuDone[t] < 2
\*   a contract is in progress while the handle its plan was created with
\*   (H1) is already destroyed (the plan outlives its handle):
NeverPlanOutlivesHandle ==
    ~(1 \in Hs /\ hSt[1] = "dead" /\ \E t \in Th : pc[t] <= RetIdx(t))
\*   two threads read one plan location at the same time:
NeverConcurrentRead == \A l \in Locs : Cardinality({x[1] : x \in inacc[l]}) <= 1
\*   the GPU operations of two calls interleave on one stream:
NeverInterleaved ==
    \A s \in Streams : ~\E i, q, j \in DOMAIN queue[s] :
        /\ i < q /\ q < j
        /\ queue[s][i].t = queue[s][j].t /\ queue[s][q].t # queue[s][i].t
=============================================================================
