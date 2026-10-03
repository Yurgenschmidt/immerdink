"""immerdink host queries from Python: exact memory footprints without a GPU.

A thin ctypes wrapper over libimmerdink_host (include/immerdink/immerdink_host.h, DESIGN
§6a). Every number comes from the C library, the same code libimmerdink uses to size its
plans and workspaces; nothing is recomputed here. No CUDA, GPU or driver is needed.

    from immerdink import host
    spec, ext = "kjbc,kiac->abij", dict(a=64, b=64, c=64, i=32, j=32, k=32)
    host.workspace_size(spec, ext, N=14, g=7)        # bytes imd_contract needs
    host.workspace_layout(spec, ext, N=14)           # the §4a region table
    host.plan_bytes(spec, ext, N=14, vec_mode="RESIDENT")   # (device, host) bytes of a plan
    host.lowering(spec, ext, N=14)                   # GEMM view, dispatch, launches, sizes
    host.g_tradeoff(spec, ext, N=14)                 # workspace for every g in 1..N

Operands: an einsum-style spec with an extents dict (labels are single characters; strides
default to dense with the last mode fastest, or pass strides=(sA, sB, sC)), or raw
descriptors (A, B, C), each (extent, stride, label) with integer labels.

The library: `lib=` path, else $IMMERDINK_HOST_LIB (the .so), else
$IMMERDINK_LIB_DIR/libimmerdink_host.so, else $IMMERDINK_BUILD_DIR or <repo>/build, else the
system search path.
"""

import ctypes as C
import ctypes.util
import os
from dataclasses import dataclass

__all__ = ["ImmerdinkError", "HostLibrary", "load", "Desc", "contraction", "workspace_size",
           "workspace_layout", "plan_bytes", "lowering", "g_tradeoff", "REGIONS", "STATUS"]

API_VERSION = 1  # the IMD_HOST_API_VERSION this wrapper was written for (minimum required)

STATUS = ["SUCCESS", "INVALID_VALUE", "NOT_SUPPORTED", "ALIASING", "INSUFFICIENT_WORKSPACE",
          "MISALIGNED", "CUDA_ERROR", "ALLOC_FAILED"]
REGIONS = ["sA", "sB", "fA", "fB", "vec", "RA", "RB", "W"]
DISPATCH = ["NOOP", "EMULATE", "DIRECT", "SCALE"]
_ROUNDING = {"RN": 0, "TRUNC": 1}
_MODE = {"EMULATE": 0, "DIRECT": 1}
_VEC = {"GENERATE": 0, "RESIDENT": 1}


class ImmerdinkError(RuntimeError):
    """A non-SUCCESS status from libimmerdink_host; .status is its name."""

    def __init__(self, status, what=""):
        self.status = STATUS[status] if 0 <= status < len(STATUS) else f"<{status}>"
        super().__init__(f"{what}: IMD_STATUS_{self.status}" if what else self.status)


# ------------------------------------------------------------------------------------------
# C structs (immerdink_host.h)

class _Tensor(C.Structure):
    _fields_ = [("rank", C.c_int32), ("extent", C.POINTER(C.c_int64)),
                ("stride", C.POINTER(C.c_int64)), ("label", C.POINTER(C.c_int32))]


class _Options(C.Structure):
    _fields_ = [("num_moduli", C.c_int32), ("residue_group", C.c_int32),
                ("rounding", C.c_int32), ("mode", C.c_int32), ("vec_mode", C.c_int32)]


class _Region(C.Structure):
    _fields_ = [("offset", C.c_uint64), ("bytes", C.c_uint64)]


class _Lowering(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("status", C.c_int32), ("lowered", C.c_int32),
                ("dispatch", C.c_int32), ("swapped", C.c_int32), ("num_moduli", C.c_int32),
                ("residue_group", C.c_int32), ("num_groups", C.c_int32),
                ("m", C.c_int64), ("n", C.c_int64), ("k", C.c_int64), ("b", C.c_int64),
                ("n_hat", C.c_int64), ("k_hat", C.c_int64), ("vec_entries", C.c_int64),
                ("k_chunks", C.c_int64), ("gemm_launches", C.c_int64),
                ("kappa", C.c_uint64), ("workspace_bytes", C.c_uint64),
                ("plan_device_bytes", C.c_uint64), ("plan_host_bytes", C.c_uint64)]


assert C.sizeof(_Lowering) == 136


# ------------------------------------------------------------------------------------------
# Library

def _candidates():
    if os.environ.get("IMMERDINK_HOST_LIB"):
        yield os.environ["IMMERDINK_HOST_LIB"]
    if os.environ.get("IMMERDINK_LIB_DIR"):
        yield os.path.join(os.environ["IMMERDINK_LIB_DIR"], "libimmerdink_host.so")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build = os.environ.get("IMMERDINK_BUILD_DIR") or os.path.join(repo, "build")
    yield os.path.join(build, "libimmerdink_host.so")
    found = ctypes.util.find_library("immerdink_host")
    if found:
        yield found


class HostLibrary:
    """libimmerdink_host, loaded with ctypes. Thread-safe (the C functions are pure)."""

    def __init__(self, path=None):
        if path is None:
            tried = []
            for cand in _candidates():
                tried.append(cand)
                if os.path.sep not in cand or os.path.exists(cand):
                    path = cand
                    break
            if path is None:
                raise OSError("libimmerdink_host not found (set IMMERDINK_HOST_LIB or "
                              f"IMMERDINK_LIB_DIR); tried {tried}")
        self.path = path
        lib = self.lib = C.CDLL(path)
        T, O = C.POINTER(_Tensor), C.POINTER(_Options)
        sig = {
            "imd_host_api_version": (C.c_uint32, []),
            "imd_workspace_size": (C.c_int, [T, T, T, O, C.POINTER(C.c_size_t)]),
            "imd_workspace_layout": (C.c_int, [T, T, T, O, C.POINTER(_Region), C.c_int32,
                                               C.POINTER(C.c_int32), C.POINTER(C.c_size_t)]),
            "imd_region_name": (C.c_char_p, [C.c_int32]),
            "imd_plan_bytes": (C.c_int, [T, T, T, O, C.POINTER(C.c_size_t),
                                         C.POINTER(C.c_size_t)]),
            "imd_lowering": (C.c_int, [T, T, T, O, C.POINTER(_Lowering)]),
            "imd_status_string": (C.c_char_p, [C.c_int]),
        }
        for name, (res, args) in sig.items():
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
        self.version = lib.imd_host_api_version()
        if self.version < API_VERSION:
            raise OSError(f"{path}: host API version {self.version} < {API_VERSION}")


_loaded = {}  # path (None: the default search) -> HostLibrary


def load(path=None):
    """The library at `path`, or the default one; each is loaded once, then cached."""
    if path not in _loaded:
        _loaded[path] = HostLibrary(path)
    return _loaded[path]


# ------------------------------------------------------------------------------------------
# Operands

@dataclass(frozen=True)
class Desc:
    """A tensor descriptor (imd_tensor_t): extents, strides (elements, any sign), labels."""
    extent: tuple
    stride: tuple
    label: tuple


def dense_strides(extent):
    """Dense strides, last mode fastest."""
    st, acc = [0] * len(extent), 1
    for d in reversed(range(len(extent))):
        st[d] = acc
        acc *= max(extent[d], 1)
    return st


def contraction(spec, extents, strides=None):
    """(A, B, C) descriptors of an einsum-style spec such as "kjbc,kiac->abij". Labels are
    single characters (their code points become the integer labels); `extents` maps each to
    its extent; `strides` is None (dense, last mode fastest) or (sA, sB, sC), each None or a
    list of per-mode strides."""
    spec = spec.replace(" ", "")
    lhs, sep, out = spec.partition("->")
    ins = lhs.split(",")
    if not sep or len(ins) != 2:
        raise ValueError(f"expected 'A,B->C', got {spec!r}")
    strides = strides or (None, None, None)
    descs = []
    for labels, st in zip((ins[0], ins[1], out), strides):
        ext = [int(extents[ch]) for ch in labels]
        st = dense_strides(ext) if st is None else [int(s) for s in st]
        if len(st) != len(ext):
            raise ValueError(f"{labels}: {len(st)} strides for {len(ext)} modes")
        descs.append(Desc(tuple(ext), tuple(st), tuple(ord(ch) for ch in labels)))
    return tuple(descs)


def _operands(spec, extents, strides):
    if isinstance(spec, str):
        if extents is None:
            raise ValueError("an einsum spec needs extents")
        return contraction(spec, extents, strides)
    if extents is not None or strides is not None:
        raise ValueError("extents / strides are only for an einsum spec")
    descs = tuple(d if isinstance(d, Desc) else Desc(*map(tuple, d)) for d in spec)
    if len(descs) != 3:
        raise ValueError("expected (A, B, C) descriptors")
    return descs


class _Call:
    """C arguments for one query; keeps the arrays alive."""

    def __init__(self, descs, N, g, rounding, mode, vec_mode):
        self.t = []
        for d in descs:
            r = len(d.extent)
            if not (len(d.stride) == len(d.label) == r):
                raise ValueError("extent, stride and label lengths differ")
            self.t.append(_Tensor(r, (C.c_int64 * r)(*d.extent), (C.c_int64 * r)(*d.stride),
                                  (C.c_int32 * r)(*d.label)))
        self.o = _Options(int(N), int(g), _ROUNDING[rounding], _MODE[mode], _VEC[vec_mode])

    def args(self):
        return [C.byref(x) for x in self.t] + [C.byref(self.o)]


def _prep(spec, extents, strides, N, g, rounding, mode, vec_mode, lib):
    return load(lib) if not isinstance(lib, HostLibrary) else lib, \
        _Call(_operands(spec, extents, strides), N, g, rounding, mode, vec_mode)


# ------------------------------------------------------------------------------------------
# Queries (DESIGN §6a). Options: N (2..20), g (1..N; 0 = N), rounding "RN" | "TRUNC",
# mode "EMULATE" | "DIRECT", vec_mode "GENERATE" | "RESIDENT".

def workspace_size(spec, extents=None, *, strides=None, N, g=0, rounding="RN",
                   mode="EMULATE", vec_mode="GENERATE", lib=None):
    """imd_workspace_size: the exact workspace bytes (DESIGN §4a)."""
    h, call = _prep(spec, extents, strides, N, g, rounding, mode, vec_mode, lib)
    out = C.c_size_t(0)
    st = h.lib.imd_workspace_size(*call.args(), C.byref(out))
    if st != 0:
        raise ImmerdinkError(st, "imd_workspace_size")
    return out.value


@dataclass(frozen=True)
class Layout:
    total: int
    regions: dict   # region name -> (offset, bytes), present regions only, §4a order

    def __str__(self):
        rows = [f"  {n:<4} offset {o:>14,}  bytes {b:>14,}" for n, (o, b) in self.regions.items()]
        return "\n".join([f"workspace {self.total:,} bytes"] + rows)


def workspace_layout(spec, extents=None, *, strides=None, N, g=0, rounding="RN",
                     mode="EMULATE", vec_mode="GENERATE", lib=None):
    """imd_workspace_layout: the §4a region table."""
    h, call = _prep(spec, extents, strides, N, g, rounding, mode, vec_mode, lib)
    regs = (_Region * len(REGIONS))()
    count, total = C.c_int32(0), C.c_size_t(0)
    st = h.lib.imd_workspace_layout(*call.args(), regs, len(REGIONS), C.byref(count),
                                    C.byref(total))
    if st != 0:
        raise ImmerdinkError(st, "imd_workspace_layout")
    names = [h.lib.imd_region_name(r).decode() for r in range(min(count.value, len(REGIONS)))]
    return Layout(total.value, {n: (regs[r].offset, regs[r].bytes)
                                for r, n in enumerate(names) if regs[r].bytes})


def plan_bytes(spec, extents=None, *, strides=None, N, g=0, rounding="RN", mode="EMULATE",
               vec_mode="GENERATE", lib=None):
    """imd_plan_bytes: (device, host) bytes a plan for these arguments would hold."""
    h, call = _prep(spec, extents, strides, N, g, rounding, mode, vec_mode, lib)
    dev, host = C.c_size_t(0), C.c_size_t(0)
    st = h.lib.imd_plan_bytes(*call.args(), C.byref(dev), C.byref(host))
    if st != 0:
        raise ImmerdinkError(st, "imd_plan_bytes")
    return dev.value, host.value


@dataclass(frozen=True)
class Lowering:
    """imd_lowering_t (field meanings: immerdink_host.h); status / dispatch as names."""
    status: str
    lowered: bool
    dispatch: str
    swapped: bool
    num_moduli: int
    residue_group: int
    num_groups: int
    m: int
    n: int
    k: int
    b: int
    n_hat: int
    k_hat: int
    vec_entries: int
    k_chunks: int
    gemm_launches: int
    kappa: int
    workspace_bytes: int
    plan_device_bytes: int
    plan_host_bytes: int


def lowering(spec, extents=None, *, strides=None, N, g=0, rounding="RN", mode="EMULATE",
             vec_mode="GENERATE", lib=None):
    """imd_lowering: the lowering summary. Never raises for a plan-time status: check
    .status ("SUCCESS", "NOT_SUPPORTED", ...) and .lowered."""
    h, call = _prep(spec, extents, strides, N, g, rounding, mode, vec_mode, lib)
    r = _Lowering()
    r.struct_size = C.sizeof(_Lowering)
    st = h.lib.imd_lowering(*call.args(), C.byref(r))
    vals = {f: getattr(r, f) for f, _ in _Lowering._fields_ if f != "struct_size"}
    vals.update(status=STATUS[st], lowered=bool(r.lowered), swapped=bool(r.swapped),
                dispatch=DISPATCH[r.dispatch])
    return Lowering(**vals)


def g_tradeoff(spec, extents=None, *, strides=None, N, rounding="RN", mode="EMULATE",
               vec_mode="GENERATE", lib=None):
    """The workspace / speed knob (DESIGN §4a, BENCHMARKS "g"): for g = 1..N, a dict with g,
    the workspace bytes, the residue groups and the K4 launches per call."""
    rows = []
    for g in range(1, int(N) + 1):
        lw = lowering(spec, extents, strides=strides, N=N, g=g, rounding=rounding, mode=mode,
                      vec_mode=vec_mode, lib=lib)
        if not lw.lowered:
            raise ImmerdinkError(STATUS.index(lw.status), "imd_lowering")
        rows.append({"g": g, "workspace": lw.workspace_bytes, "groups": lw.num_groups,
                     "gemm_launches": lw.gemm_launches})
    return rows


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog="python3 -m immerdink.host",
        description="Exact immerdink footprints (no GPU): g trade-off table for a contraction.")
    ap.add_argument("spec", help='einsum spec, e.g. "kjbc,kiac->abij"')
    ap.add_argument("extents", help="label=extent,... e.g. a=64,b=64,c=64,i=32,j=32,k=32")
    ap.add_argument("-N", type=int, default=14)
    ap.add_argument("--rounding", default="RN", choices=sorted(_ROUNDING))
    ap.add_argument("--vec-mode", default="GENERATE", choices=sorted(_VEC))
    ap.add_argument("--lib", default=None, help="path to libimmerdink_host.so")
    a = ap.parse_args(argv)
    ext = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.extents.split(",")}
    kw = dict(N=a.N, rounding=a.rounding, vec_mode=a.vec_mode, lib=a.lib)
    lw = lowering(a.spec, ext, **kw)
    print(f"{a.spec}  N={a.N}  {lw.dispatch}  m={lw.m} n={lw.n} k={lw.k} b={lw.b}"
          f"{'  (swapped)' if lw.swapped else ''}  status {lw.status}")
    dev, host = plan_bytes(a.spec, ext, **kw) if lw.status == "SUCCESS" else (0, 0)
    print(f"plan: {dev:,} device bytes, {host:,} host bytes")
    print(f"{'g':>3} {'workspace bytes':>16} {'GiB':>8} {'vs g=N':>7} {'groups':>6} {'launches':>8}")
    rows = g_tradeoff(a.spec, ext, **kw)
    full = rows[-1]["workspace"] or 1
    for r in rows:
        print(f"{r['g']:>3} {r['workspace']:>16,} {r['workspace'] / 2**30:>8.3f} "
              f"{100 * r['workspace'] / full:>6.0f}% {r['groups']:>6} {r['gemm_launches']:>8}")


if __name__ == "__main__":
    _main()
