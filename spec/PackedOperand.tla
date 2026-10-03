---------------------------- MODULE PackedOperand ----------------------------
(***************************************************************************)
(* Packed-operand reuse (DESIGN §6b, DRAFT for review; ROADMAP D1).        *)
(*                                                                         *)
(* An operand X is packed once (imd_pack) into a caller-owned buffer: its  *)
(* shifts σ, non-finite flags and all N residue planes, laid out by the    *)
(* §6b byte contract. A plan created against the pack (its compatibility   *)
(* KEY) adopts the pack's canonical K order and reads the packed bytes in  *)
(* place of packing X in-line. Two parts, two specifications:              *)
(*                                                                         *)
(* CompatSpec (constant-level; Init chooses the data, then stutters)       *)
(*   X (caller's A, one row: r = 1) carries two contracted labels k1, k2   *)
(*   of extent E1 = E2 = 2 (k = 4, k̂ = ⌈k⌉_KAlign = 8); the partner Y     *)
(*   (one column) carries k1, k2. The numerics are DESIGN §3 at toy        *)
(*   widths, in exact integers: σ = the largest u with c·S·4^u ≤ R (§3.2   *)
(*   step 4; RN: c = 4, R = (2T − κ)², κ = ⌈√k⌉; TRUNC: c = 1, R = T²;    *)
(*   integer data, so S = Σx² is exact and γ = 0), RN-even / truncating    *)
(*   quantization (§3.3), SMod residues (§3.4), per-plane int dot mod p_l  *)
(*   over k̂ (§3.5), CRT (§3.6), output z·2^−(σ+τ) compared as an exact    *)
(*   rational, and the native path for flagged outputs (§3.7) abstracted   *)
(*   as the ORDERED sequence of (a_h, b_h) pairs (the fma chain is order-  *)
(*   sensitive: oracle A7).                                                *)
(*   The packed buffer is a byte array laid out as §6b:                    *)
(*     res[o], o = (l−1)·b·r·k̂ + (λ·r + ρ)·k̂ + h,  then a zeroed tail of  *)
(*     (⌈r⌉_RowAlign − r)·k̂ bytes;  σ and the flag in their own regions.  *)
(*   The plan reads it as K4 would: post-swap role A reads rows ρ < m = r; *)
(*   role B reads ρ < n̂ = ⌈r⌉_RowAlign (the GEMM's padded N; rows ≥ r     *)
(*   only feed W columns ≥ n, which K5 never reads, so they need only be   *)
(*   in bounds), planes read group by group (g ∈ Gs).                      *)
(*   Checked (with the key check, or with the pack built to the key):      *)
(*     SigmaEqual, WEqual, CEqual  the packed contraction reads the same   *)
(*                 σ, produces the same W planes and the same C element as *)
(*                 packing in-line with the plan's (adopted) K order, for  *)
(*                 every post-swap role and every residue-group size g;    *)
(*     NoOOBRead   every packed byte the GEMM reads is inside the buffer;  *)
(*     EmulatedOrderFree  for unflagged outputs the in-line result is the  *)
(*                 same for both K orders: adopting the pack's order       *)
(*                 changes nothing for finite data;                        *)
(*     AcceptedWhenKeyMatches  a pack built to the plan's key is accepted. *)
(*   Mutations (one per negative model; "none" = the design):              *)
(*     key components (bound without the key check, Cmp.check = FALSE):   *)
(*       "order"   the plan keeps its own default K order (which in v1.5   *)
(*                 depends on the partner and C, §5 step 8) instead of     *)
(*                 adopting the pack's                                     *)
(*       "N"       the pack was made with N + 1 moduli                     *)
(*       "rnd"     the pack was made with the other rounding               *)
(*       "kset"    the pack's K set lacks k2 (k2 is a partner-only label:  *)
(*                 the plan sums over it with stride 0 in X)               *)
(*       "kalign"  the pack's k̂ alignment (format) differs: 2·KAlign       *)
(*       "partnerOrder"  both operands packed, the partner's pack has the  *)
(*                 other K order                                           *)
(*     byte-contract components (not key fields: every pack of a format    *)
(*     has them):                                                          *)
(*       "brole"   in the post-swap B role the plan reads the pack with   *)
(*                 the v1 workspace RB strides (n̂ rows per block)          *)
(*       "notail"  the pack has no zeroed tail                             *)
(*       "nopad"   both packs leave their K padding unwritten (junk)       *)
(*       "noflags" the pack does not store the non-finite flags            *)
(*   With Cmp.check = TRUE the plan compares keys and returns              *)
(*   PACK_MISMATCH instead of contracting (pos_key_check: every key        *)
(*   mutation is rejected, none gives a wrong result).                     *)
(*                                                                         *)
(* StreamSpec (state machine): the caller's ordering of imd_pack, the      *)
(* contracts that read the packed buffer and the FP64 operand (native      *)
(* path), and a later write of either (a re-pack with new values, a free,  *)
(* an overwrite of the FP64 operand), on streams with optional event       *)
(* edges. Each operation is NC CTAs, one cell each, interleaved.           *)
(*     NoStaleRead     every contract reads the packed cells and the FP64  *)
(*                     cells of the version it was issued against.         *)
(*   imd_pack writes are visible to a contract only through stream order   *)
(*   or an event (RAW); a later write must wait for the readers (WAR).     *)
(*   Both are the caller's job: immerdink owns no streams or events.       *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Cmp,    \* compatibility scenario: [N, rnd, roles, gs, defOrder, both,
            \*   muts, check, vx, vy]  (MC_PackedOperand.tla)
    Strm    \* stream scenario: [pack, c1, c2, postSt, post, raw, war]

(* ----------------------------------------------------------------------- *)
(* Toy parameters (real: moduli 256, 255, 253, ...; k̂ = ⌈k⌉₁₆; n̂ = ⌈n⌉₁₆) *)
(* ----------------------------------------------------------------------- *)
Mods     == <<8, 7, 5, 3>>       \* pairwise coprime, first one even (as 256)
E1       == 2
E2       == 2
KAlign   == 8                    \* k̂ = ⌈k⌉_KAlign: k = 4 -> k̂ = 8 (4 padding)
RowAlign == 2                    \* n̂ = ⌈n⌉_RowAlign: r = 1 -> n̂ = 2
INF      == 99                   \* a non-finite FP64 entry (Inf or NaN)
JUNK     == 1                    \* an unwritten int8 / an out-of-bounds read
KP       == E1 * E2              \* the plan's k
Ceil(x, a) == ((x + a - 1) \div a) * a
KH       == Ceil(KP, KAlign)     \* the plan's k̂
RowHat   == Ceil(1, RowAlign)    \* n̂ for r = 1
Idx      == (0..E1-1) \X (0..E2-1)

RECURSIVE ProdTo(_)
ProdTo(n) == IF n = 0 THEN 1 ELSE Mods[n] * ProdTo(n - 1)
CeilSqrt(x) == CHOOSE c \in 0..x : c * c >= x /\ (c = 0 \/ (c - 1) * (c - 1) < x)
\* constant tables (no variables: TLC evaluates them once)
Ptab  == [n \in 1..Len(Mods) |-> ProdTo(n)]
Ttab  == [n \in 1..Len(Mods) |->                         \* T = ⌊√H⌋, H = P/2 − 1
            CHOOSE t \in 0..64 : t * t <= Ptab[n] \div 2 - 1
                                 /\ (t + 1) * (t + 1) > Ptab[n] \div 2 - 1]
Qtab  == [n \in 1..Len(Mods) |->                         \* q_l = M_l⁻¹ mod p_l
            [l \in 1..n |-> CHOOSE q \in 1..Mods[l] - 1 :
                               ((Ptab[n] \div Mods[l]) * q) % Mods[l] = 1]]
Kappa == [x \in 0..16 |-> CeilSqrt(x)]
P(n) == Ptab[n]
T(n) == Ttab[n]

RECURSIVE SumF(_, _)
SumF(f, n) == IF n = 0 THEN 0 ELSE f[n - 1] + SumF(f, n - 1)

ASSUME \A i, j \in 1..Len(Mods) : i # j =>                 \* pairwise coprime
           \A d \in 2..8 : ~(Mods[i] % d = 0 /\ Mods[j] % d = 0)
ASSUME Cmp.N + 1 <= Len(Mods) /\ Cmp.N >= 2
ASSUME \A n \in {Cmp.N, Cmp.N + 1} : Kappa[KP] < T(n)   \* §3.2 RN acceptance holds

(* §3.1-§3.3 for one row (raw: k̄ -> value, INF allowed) *)
Fin(v) == IF v = INF THEN 0 ELSE v
URange == -6..6
RuleOK(L, R, u) == IF u >= 0 THEN L * (4 ^ u) <= R ELSE L <= R * (4 ^ (-u))
Sigma(xs, k, n, rnd) ==
    LET S == SumF([h \in 0..k-1 |-> xs[h] * xs[h]], k)
        L == IF rnd = "RN" THEN 4 * S ELSE S
        R == IF rnd = "RN" THEN (2 * T(n) - Kappa[k]) ^ 2 ELSE T(n) ^ 2
    IN IF S = 0 THEN 0
       ELSE CHOOSE u \in URange : RuleOK(L, R, u) /\ ~RuleOK(L, R, u + 1)
Quant(x, s, rnd) ==
    IF s >= 0 THEN x * (2 ^ s)
    ELSE LET d   == 2 ^ (-s)
             a   == IF x < 0 THEN -x ELSE x
             q   == a \div d
             rem == a % d
             r   == IF rnd = "RN" /\ (2 * rem > d \/ (2 * rem = d /\ q % 2 = 1))
                    THEN q + 1 ELSE q
         IN IF x < 0 THEN -r ELSE r
SMod(x, p) == ((x + p \div 2) % p) - p \div 2
Scale(raw, k, n, rnd) ==
    LET xs == [h \in 0..k-1 |-> Fin(raw[h])]
        s  == Sigma(xs, k, n, rnd)
    IN [sig |-> s, flag |-> \E h \in 0..k-1 : raw[h] = INF,
        q |-> [h \in 0..k-1 |-> Quant(xs[h], s, rnd)]]

(* canonical k̄ -> (i1, i2) under a K order; "12" = k1 fastest *)
Dec(order, h) == IF order = "12" THEN <<h % E1, h \div E1>> ELSE <<h \div E2, h % E2>>
Other(o)      == IF o = "12" THEN "21" ELSE "12"
OtherRnd(r)   == IF r = "RN" THEN "TRUNC" ELSE "RN"

(* ----------------------------------------------------------------------- *)
(* Packs (imd_pack) and their keys                                         *)
(* ----------------------------------------------------------------------- *)
PackOrderX == "12"   \* X's AUTO order: by X's own strides (k1 fastest in X)
XPack(mut) ==
    [n |-> IF mut = "N" THEN Cmp.N + 1 ELSE Cmp.N,
     rnd |-> IF mut = "rnd" THEN OtherRnd(Cmp.rnd) ELSE Cmp.rnd,
     order |-> PackOrderX,
     k2 |-> mut # "kset",
     kalign |-> IF mut = "kalign" THEN 2 * KAlign ELSE KAlign,
     zeroPad |-> mut # "nopad", tail |-> mut # "notail", flags |-> mut # "noflags"]
\* the plan's K order: adopted from the (caller's A) pack, or v1's default
PO(mut) == IF mut = "order" THEN Cmp.defOrder ELSE XPack(mut).order
YPack(mut) ==
    [n |-> Cmp.N, rnd |-> Cmp.rnd,
     order |-> IF mut = "partnerOrder" THEN Other(PO(mut)) ELSE PO(mut),
     k2 |-> TRUE, kalign |-> KAlign,
     zeroPad |-> mut # "nopad", tail |-> TRUE, flags |-> TRUE]

PackK(pc)     == IF pc.k2 THEN KP ELSE E1
PackKh(pc)    == Ceil(PackK(pc), pc.kalign)
PackPlane(pc) == PackKh(pc)                                   \* b·r·k̂ (b = r = 1)
PackSize(pc)  == pc.n * PackPlane(pc) + (IF pc.tail THEN (RowHat - 1) * PackKh(pc) ELSE 0)
PackRaw(pc, D) == [h \in 0..PackK(pc)-1 |->
                      IF pc.k2 THEN D[Dec(pc.order, h)] ELSE D[<<h, 0>>]]
Pack(pc, D) ==
    LET sc == Scale(PackRaw(pc, D), PackK(pc), pc.n, pc.rnd)
    IN [sig |-> sc.sig, flag |-> pc.flags /\ sc.flag, size |-> PackSize(pc),
        res |-> [o \in 0..PackSize(pc)-1 |->
                    LET l == o \div PackPlane(pc)
                        h == (o % PackPlane(pc)) % PackKh(pc)
                    IN IF l >= pc.n THEN 0                          \* the tail
                       ELSE IF h < PackK(pc) THEN SMod(sc.q[h], Mods[l + 1])
                       ELSE IF pc.zeroPad THEN 0 ELSE JUNK]]

\* The compatibility key (§6b): N, rounding, the K set (here: k2 or not),
\* the format (here: the k̂ alignment), the K order. (The FP64 element map,
\* the row and batch classes, extents and strides are fixed in this toy.)
Key(pc)      == [n |-> pc.n, rnd |-> pc.rnd, k2 |-> pc.k2, fmt |-> pc.kalign,
                 order |-> pc.order]
SlotKey(mut) == [n |-> Cmp.N, rnd |-> Cmp.rnd, k2 |-> TRUE, fmt |-> KAlign,
                 order |-> PO(mut)]
Status(mut) ==
    IF ~Cmp.check THEN "SUCCESS"
    ELSE IF Key(XPack(mut)) # SlotKey(mut)
            \/ (Cmp.both /\ Key(YPack(mut)) # SlotKey(mut))
         THEN "PACK_MISMATCH" ELSE "SUCCESS"

(* ----------------------------------------------------------------------- *)
(* Contraction: in-line reference and packed path                          *)
(* ----------------------------------------------------------------------- *)
VARIABLES mut, XD, YD,                    \* CompatSpec
          opSt, ctaDone, pbuf, xbuf, seen \* StreamSpec

RowOf(D, order) == [h \in 0..KP-1 |-> D[Dec(order, h)]]
InRes(sc, l)    == [h \in 0..KH-1 |-> IF h < KP THEN SMod(sc.q[h], Mods[l]) ELSE 0]
DotMod(a, b, p) == SumF([h \in 0..KH-1 |-> a[h] * b[h]], KH) % p
\* §3.6: x = Σ W_l·M_l·q_l mod P, centered to [−P/2, P/2 − 1]
CRT(W) == LET n == Cmp.N
              x == SumF([i \in 0..n-1 |-> W[i + 1] * (P(n) \div Mods[i + 1]) * Qtab[n][i + 1]], n)
                   % P(n)
          IN IF x <= P(n) \div 2 - 1 THEN x ELSE x - P(n)
Native(order) == [h \in 0..KP-1 |-> <<XD[Dec(order, h)], YD[Dec(order, h)]>>]
Out(sx, fx, sy, fy, W, order) ==
    IF fx \/ fy THEN [kind |-> "native", seq |-> Native(order)]
    ELSE [kind |-> "crt", z |-> CRT(W), s |-> sx + sy]
\* z·2^−s as a reduced dyadic: equal values have equal normal forms
RECURSIVE Norm(_, _)
Norm(z, s) == IF z = 0 THEN <<0, 0>>
              ELSE IF z % 2 = 0 THEN Norm(z \div 2, s - 1) ELSE <<z, s>>
Same(a, b) ==
    /\ a.kind = b.kind
    /\ IF a.kind = "native" THEN a.seq = b.seq ELSE Norm(a.z, a.s) = Norm(b.z, b.s)

\* in-line: both operands scaled and packed by this call, in the plan's order
Inline(order) ==
    LET x == Scale(RowOf(XD, order), KP, Cmp.N, Cmp.rnd)
        y == Scale(RowOf(YD, order), KP, Cmp.N, Cmp.rnd)
        W == [l \in 1..Cmp.N |-> DotMod(InRes(x, l), InRes(y, l), Mods[l])]
    IN [x |-> x, y |-> y, W |-> W, out |-> Out(x.sig, x.flag, y.sig, y.flag, W, order)]

\* packed: how the plan reads a packed slot (K4 operand), post-swap role
Rows(role) == IF role = "A" THEN 0..0 ELSE 0..RowHat-1
BlockStride(m, role, isX) ==
    IF isX /\ role = "B" /\ m = "brole" THEN RowHat * KH ELSE KH
\* plane l = (gi − 1)·g + s: slot s of residue group gi (g moduli per group)
Groups(g)  == 1..((Cmp.N + g - 1) \div g)
Slots(g, gi) == 1..(IF gi * g <= Cmp.N THEN g ELSE Cmp.N - (gi - 1) * g)
Off(m, role, isX, g, gi, s, rho, h) ==
    (((gi - 1) * g + s) - 1) * BlockStride(m, role, isX) + rho * KH + h
Rd(buf, o) == IF o < buf.size THEN buf.res[o] ELSE JUNK
SlotRow(buf, m, role, isX, g, gi, s) ==
    [h \in 0..KH-1 |-> Rd(buf, Off(m, role, isX, g, gi, s, 0, h))]
Offsets(m, role, isX, g) ==
    UNION {{Off(m, role, isX, g, gi, s, rho, h) :
                s \in Slots(g, gi), rho \in Rows(role), h \in 0..KH-1} : gi \in Groups(g)}
Partner(role) == IF role = "A" THEN "B" ELSE "A"

\* everything one state needs, computed once: the packs, the in-line reference
Ctx(m) ==
    LET po == PO(m)
        xb == Pack(XPack(m), XD)
        yb == IF Cmp.both THEN Pack(YPack(m), YD) ELSE [sig |-> 0, flag |-> FALSE, size |-> 0, res |-> <<>>]
        il == Inline(po)
    IN [m |-> m, po |-> po, xb |-> xb, yb |-> yb, il |-> il]
\* W plane l through the group schedule of g
PackedW(c, role, g) ==
    [l \in 1..Cmp.N |->
        LET gi == (l - 1) \div g + 1
            s  == l - (gi - 1) * g
            xr == SlotRow(c.xb, c.m, role, TRUE, g, gi, s)
            yr == IF Cmp.both THEN SlotRow(c.yb, c.m, Partner(role), FALSE, g, gi, s)
                  ELSE InRes(c.il.y, l)
        IN DotMod(xr, yr, Mods[l])]
PackedOut(c, W) ==
    LET sy == IF Cmp.both THEN c.yb.sig ELSE c.il.y.sig
        fy == IF Cmp.both THEN c.yb.flag ELSE c.il.y.flag
    IN Out(c.xb.sig, c.xb.flag, sy, fy, W, c.po)

(* ----------------------------------------------------------------------- *)
(* CompatSpec                                                               *)
(* ----------------------------------------------------------------------- *)
StreamOps == <<"pack", "c1", "c2", "post">>          \* host issue order
NC        == 2
Cells     == 1..NC
StreamInit0 ==
    /\ opSt = [o \in {"pack", "c1", "c2", "post"} |-> "wait"]
    /\ ctaDone = [o \in {"pack", "c1", "c2", "post"} |-> {}]
    /\ pbuf = [c \in Cells |-> "junk"]
    /\ xbuf = [c \in Cells |-> "x1"]
    /\ seen = [o \in {"pack", "c1", "c2"} |-> [c \in Cells |-> <<"none", "none">>]]

XData(m) == IF m = "kset"   \* X does not carry k2: constant along it (stride 0)
            THEN {f \in [Idx -> Cmp.vx] : \A i \in Idx : f[i] = f[<<i[1], 0>>]}
            ELSE [Idx -> Cmp.vx]
CompatInit ==
    /\ mut \in Cmp.muts
    /\ XD \in XData(mut)
    /\ YD \in [Idx -> Cmp.vy]
    /\ StreamInit0
vars == <<mut, XD, YD, opSt, ctaDone, pbuf, xbuf, seen>>
CompatSpec == CompatInit /\ [][UNCHANGED vars]_vars

OK(m) == Status(m) = "SUCCESS"
SigmaEqual == OK(mut) => LET c == Ctx(mut) IN c.xb.sig = c.il.x.sig
WEqual     == OK(mut) => LET c == Ctx(mut) IN
                 \A role \in Cmp.roles, g \in Cmp.gs : PackedW(c, role, g) = c.il.W
CEqual     == OK(mut) => LET c == Ctx(mut) IN
                 \A role \in Cmp.roles, g \in Cmp.gs :
                     Same(PackedOut(c, PackedW(c, role, g)), c.il.out)
NoOOBRead  == OK(mut) => LET c == Ctx(mut) IN
                 \A role \in Cmp.roles, g \in Cmp.gs :
                    /\ Offsets(mut, role, TRUE, g) \subseteq 0..c.xb.size-1
                    /\ Cmp.both => Offsets(mut, Partner(role), FALSE, g)
                                       \subseteq 0..c.yb.size-1
EmulatedOrderFree ==
    LET a == Inline("12")
        b == Inline("21")
    IN ~a.x.flag /\ ~a.y.flag => Same(a.out, b.out)
AcceptedWhenKeyMatches == mut = "none" => OK(mut)
\* witnesses (expected to FAIL on positive configurations)
NeverMismatch   == OK(mut)
NativeOrderFree == Same(Inline("12").out, Inline("21").out)


(* ----------------------------------------------------------------------- *)
(* StreamSpec                                                               *)
(* ----------------------------------------------------------------------- *)
\* Strm: stream of each op (pack, c1, c2, postSt), the later write
\* (post: "none" | "repack" | "free" | "writeX"), event edges raw (pack ->
\* each contract) and war (both contracts -> post).
Ops == {"pack", "c1", "c2"} \cup (IF Strm.post = "none" THEN {} ELSE {"post"})
Pos(o) == CHOOSE i \in 1..4 : StreamOps[i] = o
StOf(o) == CASE o = "pack" -> Strm.pack [] o = "c1" -> Strm.c1
             [] o = "c2" -> Strm.c2   [] o = "post" -> Strm.postSt
Deps(o) == CASE o \in {"c1", "c2"} -> (IF Strm.raw THEN {"pack"} ELSE {})
             [] o = "post" -> (IF Strm.war THEN {"c1", "c2"} ELSE {})
             [] OTHER -> {}
Ready(o) ==
    /\ \A o2 \in Ops : Pos(o2) < Pos(o) /\ StOf(o2) = StOf(o) => opSt[o2] = "done"
    /\ \A d \in Deps(o) : opSt[d] = "done"
Start(o) == /\ o \in Ops /\ opSt[o] = "wait" /\ Ready(o)
            /\ opSt' = [opSt EXCEPT ![o] = "run"]
            /\ UNCHANGED <<mut, XD, YD, ctaDone, pbuf, xbuf, seen>>
\* one CTA: the pack reads X's cell and writes the packed cell; a contract
\* reads the packed cell (K4/K5) and X's cell (K5's native path); post writes
Cta(o, c) ==
    /\ o \in Ops /\ opSt[o] = "run" /\ c \notin ctaDone[o]
    /\ ctaDone' = [ctaDone EXCEPT ![o] = @ \cup {c}]
    /\ CASE o = "pack" ->
              /\ pbuf' = [pbuf EXCEPT ![c] = "v1"]
              /\ seen' = [seen EXCEPT ![o][c] = <<"v1", xbuf[c]>>]
              /\ UNCHANGED xbuf
         [] o \in {"c1", "c2"} ->
              /\ seen' = [seen EXCEPT ![o][c] = <<pbuf[c], xbuf[c]>>]
              /\ UNCHANGED <<pbuf, xbuf>>
         [] o = "post" ->
              /\ CASE Strm.post = "repack" -> pbuf' = [pbuf EXCEPT ![c] = "v2"] /\ UNCHANGED xbuf
                   [] Strm.post = "free"   -> pbuf' = [pbuf EXCEPT ![c] = "freed"] /\ UNCHANGED xbuf
                   [] Strm.post = "writeX" -> xbuf' = [xbuf EXCEPT ![c] = "x2"] /\ UNCHANGED pbuf
              /\ UNCHANGED seen
    /\ UNCHANGED <<mut, XD, YD, opSt>>
Finish(o) == /\ o \in Ops /\ opSt[o] = "run" /\ ctaDone[o] = Cells
             /\ opSt' = [opSt EXCEPT ![o] = "done"]
             /\ UNCHANGED <<mut, XD, YD, ctaDone, pbuf, xbuf, seen>>
StreamNext == \E o \in Ops : Start(o) \/ Finish(o) \/ \E c \in Cells : Cta(o, c)
Done == \A o \in Ops : opSt[o] = "done"
StreamInit ==
    /\ mut = "none"
    /\ XD = [i \in Idx |-> 1]
    /\ YD = [i \in Idx |-> 1]
    /\ StreamInit0
StreamSpec == StreamInit /\ [][StreamNext \/ (Done /\ UNCHANGED vars)]_vars
              /\ WF_vars(StreamNext)

\* every contract reads the packed version v1 and the FP64 values x1 it was
\* packed from (the native path must see the same X as the pack)
NoStaleRead == \A o \in {"c1", "c2"}, c \in Cells :
                   seen[o][c] \in {<<"none", "none">>, <<"v1", "x1">>}
PackReadsX1 == \A c \in Cells : seen["pack"][c] \in {<<"none", "none">>, <<"v1", "x1">>}
CompleteAtEnd == Done => \A o \in {"c1", "c2"}, c \in Cells : seen[o][c] = <<"v1", "x1">>
Termination == <>Done
\* witness: two contracts read one packed buffer concurrently
NeverConcurrentReads == ~(opSt["c1"] = "run" /\ opSt["c2"] = "run")
=============================================================================
