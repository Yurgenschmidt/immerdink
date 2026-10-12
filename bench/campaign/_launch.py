"""Exec helper of the runner: make the experiment die with the runner (Linux PR_SET_PDEATHSIG), then exec it.

    python3 -I _launch.py <argv...>

A runner killed with kill -9 (or a Slurm job cancelled outside its cgroup) must not leave an experiment writing into a
folder the next run wipes. Used instead of Popen(preexec_fn=...), which is unsafe next to the runner's copy-back thread.
"""
import ctypes
import os
import signal
import sys

try:
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(signal.SIGKILL))   # PR_SET_PDEATHSIG
except OSError:
    pass
if os.getppid() == 1:        # the runner is already gone
    os._exit(1)
os.execvp(sys.argv[1], sys.argv[1:])
