"""ctxphase.py <sqlite> <first-kernel substring> [proc]: per call (a call starts at the first
kernel), the offsets of our context's switch-outs and the length of the resident slice before.
Input as for ctxstat.py. docs/NOTES-K5-bimodal.md."""
import sqlite3
import sys

c = sqlite3.connect(sys.argv[1])
procs = {gp: name for gp, pid, name in c.execute("select globalPid,pid,name from PROCESSES")}
rows = list(c.execute("select tag,contextId,timestamp,globalPid from GPU_CONTEXT_SWITCH_EVENTS order by timestamp"))
k = list(c.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                   "join StringIds s on k.shortName=s.id order by k.start"))
me = sys.argv[3] if len(sys.argv) > 3 else [n for n in set(procs.get(r[3]) for r in rows) if n and ('probe' in n or 'bench' in n)][0]
starts = [a for a, b, n in k if sys.argv[1 + 1] in n]
ends = []
for i, s in enumerate(starts):
    nxt = starts[i + 1] if i + 1 < len(starts) else 1 << 62
    ends.append(max(b for a, b, n in k if s <= a < nxt))
mine = [(tag, ts) for tag, ctx, ts, gp in rows if procs.get(gp) == me]
for s, e in zip(starts, ends):
    restore_before = max([ts for tag, ts in mine if tag == 8 and ts <= s] or [0])
    outs = [ts for tag, ts in mine if tag == 7 and s <= ts <= e]
    seq = ' '.join('%.2f' % ((t - s) / 1e6) for t in outs)
    sl = ' '.join('%.2f' % ((t - r) / 1e6) for t, r in zip(outs, [restore_before] + [ts for tag, ts in mine if tag == 8 and s <= ts <= e]))
    print('call %.3f ms: last restore %.2f ms before start; switched out at +[%s] ms; slice lengths [%s] ms' % (
        (e - s) / 1e6, (s - restore_before) / 1e6, seq, sl))
