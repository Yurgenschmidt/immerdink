"""ctxstat.py <nsys sqlite> [process-name substring]: per kernel name, duration vs. the number of
times our GPU context was switched out (time-slice preemption) during the kernel, the off-GPU gap
(our SAVE_END -> our next RESTORE_START) and the residual (extra time beyond the gap).
The report must come from `nsys profile --gpuctxsw=true --trace=cuda` exported with
`nsys export --type sqlite`. docs/NOTES-K5-bimodal.md."""
import collections
import sqlite3
import statistics
import sys

c = sqlite3.connect(sys.argv[1])
me = sys.argv[2] if len(sys.argv) > 2 else None
procs = {gp: name for gp, pid, name in c.execute("select globalPid,pid,name from PROCESSES")}
rows = list(c.execute("select tag,contextId,timestamp,globalPid from GPU_CONTEXT_SWITCH_EVENTS order by timestamp"))
k = list(c.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                   "join StringIds s on k.shortName=s.id order by k.start"))
# our context ids: the context whose resident intervals cover our kernels the most
names = collections.Counter(procs.get(gp, '?') for _, _, _, gp in rows)
me = [n for n in names if n and (me in n if me else ('probe' in n or 'bench' in n))][0]
# off intervals of our process: SAVE_END(ours) -> next RESTORE_START(ours)
off = []
last_save = None
for tag, ctx, ts, gp in rows:
    if procs.get(gp) != me:
        continue
    if tag == 7:
        last_save = ts
    elif tag == 8 and last_save is not None:
        off.append((last_save, ts))
        last_save = None
by = collections.defaultdict(list)
for a, b, n in k:
    gaps = [(max(a, s), min(b, e)) for s, e in off if min(b, e) > max(a, s)]
    by[n].append(((b - a) / 1e3, len(gaps), sum(e - s for s, e in gaps) / 1e3))
print('process', me, 'off-GPU intervals', len(off))
for n, l in by.items():
    if len(l) < 5:
        continue
    base = [d for d, ns, g in l if ns == 0]
    b0 = statistics.median(base) if base else float('nan')
    print('%-30s calls %3d  clean %3d median %.1f us' % (n[:30], len(l), len(base), b0))
    for ns in sorted(set(x[1] for x in l)):
        if ns == 0:
            continue
        sel = [x for x in l if x[1] == ns]
        d = statistics.median(x[0] for x in sel)
        g = statistics.median(x[2] for x in sel)
        print('    switched out %d x: %3d calls  median dur %.1f us (+%.1f), off-GPU gap %.1f us, '
              'residual %.1f us' % (ns, len(sel), d, d - b0, g, d - b0 - g))
