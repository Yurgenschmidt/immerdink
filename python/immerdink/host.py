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
    pa = host.pack_desc_for(spec, ext, operand=0, N=14)   # pack the caller's A (DESIGN §6b)
    host.pack_layout(pa)                             # its packed bytes and imd_pack workspace
    host.workspace_size(spec, ext, N=14, packing=(pa, None))  # the workspace with A packed
    host.predict(spec, ext, N=14)                    # predicted time per phase (DESIGN §6c)
    host.rank(spec, ext, N=[8, 14], budget=2**30)    # configurations ranked by predicted time

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
           "workspace_layout", "plan_bytes", "lowering", "g_tradeoff", "PackSpec", "pack_layout",
           "pack_desc_for", "REGIONS", "STATUS", "PACK_MISMATCH", "Calibration", "predict",
           "predict_pack", "rank", "param_names", "PERF_CLASSES", "PERF_PHASES"]

API_VERSION = 6  # the IMD_HOST_API_VERSION this wrapper was written for (minimum required)

STATUS = ["SUCCESS", "INVALID_VALUE", "NOT_SUPPORTED", "ALIASING", "INSUFFICIENT_WORKSPACE",
          "MISALIGNED", "CUDA_ERROR", "ALLOC_FAILED", "PACK_MISMATCH"]
# imd_lowering_t.pack_mismatch reasons (IMD_PACK_MISMATCH_*, DESIGN v1.7 §6b.5)
PACK_MISMATCH = ["NONE", "FORMAT", "MODULI", "ROUNDING", "CLASSES", "EXTENTS", "STRIDES", "ORDER"]
_ORDER = {"AUTO": 0, "GIVEN": 1}
REGIONS = ["sA", "sB", "fA", "fB", "vec", "RA", "RB", "W"]
DISPATCH = ["NOOP", "EMULATE", "DIRECT", "SCALE"]
_ROUNDING = {"RN": 0, "TRUNC": 1}
_MODE = {"EMULATE": 0, "DIRECT": 1}
_VEC = {"GENERATE": 0, "RESIDENT": 1}
K4_CONFIGS = ["128x256", "128x128", "64x128", "64x64"]  # IMD_K4_* tiles (DESIGN v1.5 §4)
# Performance model (DESIGN v1.8 §6c): IMD_PERF_CLASS_* key prefixes, IMD_PERF_PHASE_* names
PERF_CLASSES = ["k0", "kf", "kl", "pack", "k4_128x256", "k4_128x128", "k4_64x128", "k4_64x64",
                "k5", "direct", "beta", "call"]
PERF_PHASES = ["K0 vec", "KF fused", "KL A", "KL B", "K2 pack A", "K3 pack B", "K4 gemm",
               "K5 crt", "DIRECT", "K5 beta-only", "call", "pack prep", "pack KL"]
_PERF_MAX_PARAMS, _PERF_PARAM_CAPACITY, _PERF_MAX_PHASES = 6, 96, 16
_PERF_BETA_NONZERO, _PERF_ALPHA_ZERO = 1, 2
_PERF_VEC = {"GENERATE": 1, "RESIDENT": 2}
_PERF_PACK = {"NONE": 1, "A": 2, "B": 4, "AB": 8}


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
                ("rounding", C.c_int32), ("mode", C.c_int32), ("vec_mode", C.c_int32),
                ("k4_tune", C.c_int32)]


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
                ("plan_device_bytes", C.c_uint64), ("plan_host_bytes", C.c_uint64),
                ("fused", C.c_int32), ("reserved", C.c_int32), ("kernel_launches", C.c_int64),
                ("sm_count", C.c_int32), ("k4_config", C.c_int32), ("k4_tune_bytes", C.c_uint64),
                ("fused_large", C.c_int32), ("reserved4", C.c_int32),
                ("packed", C.c_int32), ("pack_mismatch", C.c_int32)]


class _PackDesc(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("num_moduli", C.c_int32), ("rounding", C.c_int32),
                ("order", C.c_int32), ("tensor", C.POINTER(_Tensor)),
                ("num_contracted", C.c_int32), ("contracted", C.POINTER(C.c_int32)),
                ("num_batch", C.c_int32), ("batch", C.POINTER(C.c_int32)),
                ("num_rows", C.c_int32), ("rows", C.POINTER(C.c_int32))]


class _PackLayout(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("status", C.c_int32), ("format", C.c_int32),
                ("num_moduli", C.c_int32), ("rounding", C.c_int32), ("reserved", C.c_int32),
                ("b", C.c_int64), ("r", C.c_int64), ("k", C.c_int64), ("k_hat", C.c_int64),
                ("kappa", C.c_uint64), ("sig", _Region), ("flag", _Region), ("res", _Region),
                ("pack_bytes", C.c_uint64), ("pack_ws_bytes", C.c_uint64),
                ("key_digest", C.c_uint64)]


class _Packing(C.Structure):
    _fields_ = [("A", C.POINTER(_PackDesc)), ("B", C.POINTER(_PackDesc))]


class _PerfCalib(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("format", C.c_int32), ("sm_count", C.c_int32),
                ("reserved", C.c_int32), ("peak_dram_bytes_per_s", C.c_double),
                ("peak_int8_ops_per_s", C.c_double), ("peak_fp64_flops_per_s", C.c_double),
                ("l2_bytes", C.c_double), ("gpu", C.c_char * 64), ("date", C.c_char * 32), ("version", C.c_char * 64),
                ("clocks", C.c_char * 128), ("notes", C.c_char * 128),
                ("param", C.c_double * _PERF_PARAM_CAPACITY)]


class _PerfRequest(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("flags", C.c_int32), ("k4_config", C.c_int32),
                ("reserved", C.c_int32)]


class _PerfPhase(C.Structure):
    _fields_ = [("phase", C.c_int32), ("cls", C.c_int32), ("launches", C.c_int64),
                ("time_s", C.c_double), ("bytes", C.c_double), ("ops", C.c_double),
                ("bound_s", C.c_double), ("x", C.c_double * _PERF_MAX_PARAMS)]


class _Perf(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("status", C.c_int32), ("dispatch", C.c_int32),
                ("k4_config", C.c_int32), ("sm_count", C.c_int32), ("num_phases", C.c_int32),
                ("time_s", C.c_double), ("bound_s", C.c_double), ("flops_eq", C.c_double),
                ("tflops_eq", C.c_double), ("kernel_launches", C.c_int64),
                ("workspace_bytes", C.c_uint64), ("plan_device_bytes", C.c_uint64),
                ("pack_bytes", C.c_uint64), ("pack_ws_bytes", C.c_uint64),
                ("phase", _PerfPhase * _PERF_MAX_PHASES)]


class _RankRequest(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("candidate_size", C.c_uint32),
                ("num_moduli_count", C.c_int32), ("residue_group_count", C.c_int32),
                ("num_moduli", C.POINTER(C.c_int32)), ("residue_groups", C.POINTER(C.c_int32)),
                ("vec_modes", C.c_int32), ("pack_modes", C.c_int32), ("reuse", C.c_int32),
                ("flags", C.c_int32), ("budget_bytes", C.c_uint64)]


class _Candidate(C.Structure):
    _fields_ = [("num_moduli", C.c_int32), ("residue_group", C.c_int32), ("vec_mode", C.c_int32),
                ("packed", C.c_int32), ("status", C.c_int32), ("feasible", C.c_int32),
                ("k4_config", C.c_int32), ("dispatch", C.c_int32), ("time_s", C.c_double),
                ("contract_s", C.c_double), ("pack_s", C.c_double), ("tflops_eq", C.c_double),
                ("workspace_bytes", C.c_uint64), ("plan_device_bytes", C.c_uint64),
                ("pack_bytes", C.c_uint64), ("pack_ws_bytes", C.c_uint64),
                ("total_bytes", C.c_uint64), ("kernel_launches", C.c_int64)]


assert C.sizeof(_Lowering) == 184  # version 5 (DESIGN §6b v1.7)
assert C.sizeof(_Options) == 24
assert C.sizeof(_PackDesc) == 72 and C.sizeof(_PackLayout) == 136
assert C.sizeof(_PerfCalib) == 1232 and C.sizeof(_PerfPhase) == 96 and C.sizeof(_Perf) == 1632
assert C.sizeof(_RankRequest) == 56 and C.sizeof(_Candidate) == 112


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
            # v5 (DESIGN §6b): packed operands
            "imd_pack_layout": (C.c_int, [C.POINTER(_PackDesc), C.POINTER(_PackLayout)]),
            "imd_pack_desc_for": (C.c_int, [T, T, T, O, C.c_int32, C.POINTER(_PackDesc),
                                            C.POINTER(C.c_int32), C.c_int32]),
            "imd_workspace_size_packed": (C.c_int, [T, T, T, O, C.POINTER(_Packing),
                                                    C.POINTER(C.c_size_t)]),
            "imd_workspace_layout_packed": (C.c_int, [T, T, T, O, C.POINTER(_Packing),
                                                      C.POINTER(_Region), C.c_int32,
                                                      C.POINTER(C.c_int32),
                                                      C.POINTER(C.c_size_t)]),
            "imd_lowering_packed": (C.c_int, [T, T, T, O, C.POINTER(_Packing),
                                              C.POINTER(_Lowering)]),
            # v6 (DESIGN §6c): the performance model
            "imd_perf_calib_default": (C.c_int, [C.POINTER(_PerfCalib)]),
            "imd_perf_calib_parse": (C.c_int, [C.c_char_p, C.c_size_t, C.POINTER(_PerfCalib),
                                               C.POINTER(C.c_int32)]),
            "imd_perf_calib_load": (C.c_int, [C.c_char_p, C.POINTER(_PerfCalib),
                                              C.POINTER(C.c_int32)]),
            "imd_perf_calib_format": (C.c_int, [C.POINTER(_PerfCalib), C.c_char_p, C.c_size_t,
                                                C.POINTER(C.c_size_t)]),
            "imd_perf_class_name": (C.c_char_p, [C.c_int32]),
            "imd_perf_param_name": (C.c_char_p, [C.c_int32, C.c_int32]),
            "imd_perf_phase_name": (C.c_char_p, [C.c_int32]),
            "imd_perf_predict": (C.c_int, [T, T, T, O, C.POINTER(_Packing),
                                           C.POINTER(_PerfRequest), C.POINTER(_PerfCalib),
                                           C.POINTER(_Perf)]),
            "imd_perf_predict_pack": (C.c_int, [C.POINTER(_PackDesc), C.POINTER(_PerfCalib),
                                                C.POINTER(_Perf)]),
            "imd_perf_rank": (C.c_int, [T, T, T, O, C.POINTER(_RankRequest),
                                        C.POINTER(_PerfCalib), C.POINTER(_Candidate), C.c_int32,
                                        C.POINTER(C.c_int32)]),
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


def _tensor(d):
    r = len(d.extent)
    if not (len(d.stride) == len(d.label) == r):
        raise ValueError("extent, stride and label lengths differ")
    return _Tensor(r, (C.c_int64 * r)(*d.extent), (C.c_int64 * r)(*d.stride),
                   (C.c_int32 * r)(*d.label))


@dataclass(frozen=True)
class PackSpec:
    """imd_pack_desc_t (DESIGN v1.7 §6b.5): the operand X (a Desc), N, rounding, the order
    ("AUTO": K and L in any order, R = the rest, each class by X's |strides|; "GIVEN": every
    label in canonical order) and the label lists K (contracted), L (batch), R (rows)."""
    tensor: Desc
    num_moduli: int
    rounding: str = "RN"
    order: str = "AUTO"
    contracted: tuple = ()
    batch: tuple = ()
    rows: tuple = ()


class _PackArg:
    """A _PackDesc and the arrays it points to."""

    def __init__(self, p):
        X = p.tensor if isinstance(p.tensor, Desc) else Desc(*map(tuple, p.tensor))
        self.t = _tensor(X)
        lists = (tuple(p.contracted), tuple(p.batch), tuple(p.rows or ()))
        self.lists = [(C.c_int32 * max(len(x), 1))(*x) for x in lists]
        self.d = _PackDesc(C.sizeof(_PackDesc), int(p.num_moduli), _ROUNDING[p.rounding],
                           _ORDER[p.order], C.pointer(self.t),
                           len(lists[0]), self.lists[0] if lists[0] else None,
                           len(lists[1]), self.lists[1] if lists[1] else None,
                           len(lists[2]), self.lists[2] if lists[2] else None)


class _Call:
    """C arguments for one query; keeps the arrays alive."""

    def __init__(self, descs, N, g, rounding, mode, vec_mode, packing=None):
        self.t = [_tensor(d) for d in descs]
        self.o = _Options(int(N), int(g), _ROUNDING[rounding], _MODE[mode], _VEC[vec_mode])
        self.packing = None
        if packing is not None:
            self.pk = [_PackArg(p) if p is not None else None for p in packing]
            self.packing = _Packing(*[C.pointer(x.d) if x is not None else None for x in self.pk])

    def args(self):
        return [C.byref(x) for x in self.t] + [C.byref(self.o)]

    def packed_args(self):
        return self.args() + [C.byref(self.packing)]


def _prep(spec, extents, strides, N, g, rounding, mode, vec_mode, lib, packing=None):
    return load(lib) if not isinstance(lib, HostLibrary) else lib, \
        _Call(_operands(spec, extents, strides), N, g, rounding, mode, vec_mode, packing)


# ------------------------------------------------------------------------------------------
# Queries (DESIGN §6a). Options: N (2..20), g (1..N; 0 = N), rounding "RN" | "TRUNC",
# mode "EMULATE" | "DIRECT", vec_mode "GENERATE" | "RESIDENT".

def workspace_size(spec, extents=None, *, strides=None, N, g=0, rounding="RN",
                   mode="EMULATE", vec_mode="GENERATE", packing=None, lib=None):
    """imd_workspace_size: the exact workspace bytes (DESIGN §4a). packing = (PackSpec or
    None, PackSpec or None) for the caller's A and B: imd_workspace_size_packed (§6b.6)."""
    h, call = _prep(spec, extents, strides, N, g, rounding, mode, vec_mode, lib, packing)
    out = C.c_size_t(0)
    if packing is None:
        st = h.lib.imd_workspace_size(*call.args(), C.byref(out))
    else:
        st = h.lib.imd_workspace_size_packed(*call.packed_args(), C.byref(out))
    if st != 0:
        raise ImmerdinkError(st, "imd_workspace_size" + ("_packed" if packing else ""))
    return out.value


@dataclass(frozen=True)
class Layout:
    total: int
    regions: dict   # region name -> (offset, bytes), present regions only, §4a order

    def __str__(self):
        rows = [f"  {n:<4} offset {o:>14,}  bytes {b:>14,}" for n, (o, b) in self.regions.items()]
        return "\n".join([f"workspace {self.total:,} bytes"] + rows)


def workspace_layout(spec, extents=None, *, strides=None, N, g=0, rounding="RN",
                     mode="EMULATE", vec_mode="GENERATE", packing=None, lib=None):
    """imd_workspace_layout: the §4a region table (packing: imd_workspace_layout_packed)."""
    h, call = _prep(spec, extents, strides, N, g, rounding, mode, vec_mode, lib, packing)
    regs = (_Region * len(REGIONS))()
    count, total = C.c_int32(0), C.c_size_t(0)
    if packing is None:
        st = h.lib.imd_workspace_layout(*call.args(), regs, len(REGIONS), C.byref(count),
                                        C.byref(total))
    else:
        st = h.lib.imd_workspace_layout_packed(*call.packed_args(), regs, len(REGIONS),
                                               C.byref(count), C.byref(total))
    if st != 0:
        raise ImmerdinkError(st, "imd_workspace_layout" + ("_packed" if packing else ""))
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
    fused: bool            # v2: the fused launch KF (K0 + K1 + group 1's packs)
    reserved: int
    kernel_launches: int   # v2: kernel launches of one imd_contract with alpha != 0
    sm_count: int          # v3: the SM count the K4 rule was evaluated for
    k4_config: int         # v3: IMD_K4_* of the plan-time rule (tile: K4_CONFIGS[k4_config])
    k4_tune_bytes: int     # v3: temporary device bytes of a k4_tune plan creation (0: none)
    fused_large: int       # v4: bit 0 / bit 1: the GEMM's A / B scaled + packed by KL
    reserved4: int
    packed: int            # v5: bit 0 / bit 1: the caller's A / B read from a pack
    pack_mismatch: str     # v5: the PACK_MISMATCH reason ("NONE" otherwise)


def lowering(spec, extents=None, *, strides=None, N, g=0, rounding="RN", mode="EMULATE",
             vec_mode="GENERATE", sm_count=0, packing=None, lib=None):
    """imd_lowering: the lowering summary. Never raises for a plan-time status: check
    .status ("SUCCESS", "NOT_SUPPORTED", "PACK_MISMATCH", ...) and .lowered. sm_count: the
    target device's SMs for the K4 configuration rule (0: the 128 of the RTX 4090).
    packing = (PackSpec or None, PackSpec or None): imd_lowering_packed (DESIGN §6b)."""
    h, call = _prep(spec, extents, strides, N, g, rounding, mode, vec_mode, lib, packing)
    r = _Lowering()
    r.struct_size = C.sizeof(_Lowering)
    r.sm_count = int(sm_count)
    if packing is None:
        st = h.lib.imd_lowering(*call.args(), C.byref(r))
    else:
        st = h.lib.imd_lowering_packed(*call.packed_args(), C.byref(r))
    vals = {f: getattr(r, f) for f, _ in _Lowering._fields_ if f != "struct_size"}
    vals.update(status=STATUS[st], lowered=bool(r.lowered), swapped=bool(r.swapped), fused=bool(r.fused),
                dispatch=DISPATCH[r.dispatch], pack_mismatch=PACK_MISMATCH[r.pack_mismatch])
    return Lowering(**vals)


@dataclass(frozen=True)
class PackLayout:
    """imd_pack_layout_t (DESIGN v1.7 §6b.4): sizes and the packed buffer's regions."""
    status: str
    format: int
    num_moduli: int
    rounding: str
    b: int
    r: int
    k: int
    k_hat: int
    kappa: int
    regions: dict        # "sig", "flag", "res" -> (offset, bytes)
    pack_bytes: int      # the packed buffer's bytes
    pack_ws_bytes: int   # imd_pack's workspace bytes
    key_digest: int


def pack_layout(pack, *, lib=None):
    """imd_pack_layout of a PackSpec. Raises ImmerdinkError for INVALID_VALUE; a NOT_SUPPORTED
    pack (kappa >= T, K empty) is returned with .status "NOT_SUPPORTED" and its sizes."""
    h = load(lib) if not isinstance(lib, HostLibrary) else lib
    arg = _PackArg(pack)
    r = _PackLayout()
    r.struct_size = C.sizeof(_PackLayout)
    st = h.lib.imd_pack_layout(C.byref(arg.d), C.byref(r))
    if st not in (0, 2):
        raise ImmerdinkError(st, "imd_pack_layout")
    return PackLayout(STATUS[st], r.format, r.num_moduli, "TRUNC" if r.rounding else "RN",
                      r.b, r.r, r.k, r.k_hat, r.kappa,
                      {n: (getattr(r, n).offset, getattr(r, n).bytes) for n in ("sig", "flag", "res")},
                      r.pack_bytes, r.pack_ws_bytes, r.key_digest)


def pack_desc_for(spec, extents=None, *, strides=None, operand, N, g=0, rounding="RN",
                  mode="EMULATE", vec_mode="GENERATE", lib=None):
    """imd_pack_desc_for: the GIVEN PackSpec of the caller's operand (0 = A, 1 = B) with the
    classes and orders the unpacked plan uses (a pack of it reproduces that contraction bit
    for bit)."""
    h, call = _prep(spec, extents, strides, N, g, rounding, mode, vec_mode, lib)
    cap = max(call.t[operand].rank, 1)
    labels = (C.c_int32 * cap)()
    out = _PackDesc()
    out.struct_size = C.sizeof(_PackDesc)
    st = h.lib.imd_pack_desc_for(*call.args(), int(operand), C.byref(out), labels, cap)
    if st != 0:
        raise ImmerdinkError(st, "imd_pack_desc_for")
    X = _operands(spec, extents, strides)[operand]
    lists = [tuple(getattr(out, f)[i] for i in range(getattr(out, "num_" + f)))
             for f in ("contracted", "batch", "rows")]
    return PackSpec(X, int(N), rounding, "GIVEN", *lists)


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


# ------------------------------------------------------------------------------------------
# Performance model (DESIGN v1.8 §6c, host API version 6)

class Calibration:
    """A performance-model calibration (imd_perf_calib_t, DESIGN §6c.2). Calibration.default()
    is the shipped one (RTX 4090); .load(path) / .parse(text) read a format-1 file; .format()
    gives its canonical text. .params maps "<class>.<feature>" to its value (ns per unit);
    .meta holds gpu, sm_count, date, version, clocks, notes and the three peaks."""

    def __init__(self, raw, lib=None):
        self._raw = raw
        self._lib = lib

    @staticmethod
    def _new():
        c = _PerfCalib()
        c.struct_size = C.sizeof(_PerfCalib)
        return c

    @classmethod
    def default(cls, lib=None):
        h = load(lib) if not isinstance(lib, HostLibrary) else lib
        c = cls._new()
        st = h.lib.imd_perf_calib_default(C.byref(c))
        if st != 0:
            raise ImmerdinkError(st, "imd_perf_calib_default")
        return cls(c, h)

    @classmethod
    def parse(cls, text, lib=None):
        """Parse format-1 text; ValueError (with the line) if it is invalid."""
        h = load(lib) if not isinstance(lib, HostLibrary) else lib
        data = text.encode() if isinstance(text, str) else bytes(text)
        c, line = cls._new(), C.c_int32(0)
        st = h.lib.imd_perf_calib_parse(data, len(data), C.byref(c), C.byref(line))
        if st != 0:
            raise ValueError(f"invalid calibration (line {line.value}): IMD_STATUS_{STATUS[st]}")
        return cls(c, h)

    @classmethod
    def load(cls, path, lib=None):
        h = load(lib) if not isinstance(lib, HostLibrary) else lib
        c, line = cls._new(), C.c_int32(0)
        st = h.lib.imd_perf_calib_load(os.fsencode(path), C.byref(c), C.byref(line))
        if st != 0:
            raise ValueError(f"{path}: invalid calibration (line {line.value}): "
                             f"IMD_STATUS_{STATUS[st]}")
        return cls(c, h)

    @classmethod
    def from_values(cls, meta, params, lib=None):
        """A calibration from a metadata dict and a {"<class>.<feature>": value} dict, through
        the format-1 text (so the C parser validates it)."""
        keys = ("gpu", "sm_count", "date", "version", "clocks", "notes",
                "peak_dram_bytes_per_s", "peak_int8_ops_per_s", "peak_fp64_flops_per_s",
                "l2_bytes")
        lines = ["format = 1"] + [f"{k} = {meta[k]}" for k in keys if k in meta]
        lines += [f"{k} = {float(v)!r}" for k, v in params.items()]
        return cls.parse("\n".join(lines) + "\n", lib)

    def format(self):
        """The canonical format-1 text (imd_perf_calib_format)."""
        h = self._lib or load()
        n = C.c_size_t(0)
        h.lib.imd_perf_calib_format(C.byref(self._raw), None, 0, C.byref(n))
        buf = C.create_string_buffer(n.value + 1)
        st = h.lib.imd_perf_calib_format(C.byref(self._raw), buf, n.value + 1, C.byref(n))
        if st != 0:
            raise ImmerdinkError(st, "imd_perf_calib_format")
        return buf.value.decode()

    @property
    def meta(self):
        r = self._raw
        return {"gpu": r.gpu.decode(), "sm_count": r.sm_count, "date": r.date.decode(),
                "version": r.version.decode(), "clocks": r.clocks.decode(),
                "notes": r.notes.decode(), "peak_dram_bytes_per_s": r.peak_dram_bytes_per_s,
                "peak_int8_ops_per_s": r.peak_int8_ops_per_s,
                "peak_fp64_flops_per_s": r.peak_fp64_flops_per_s, "l2_bytes": r.l2_bytes}

    @property
    def params(self):
        out = {}
        for cl, names in enumerate(param_names(self._lib).values()):
            for j, name in enumerate(names):
                if name is not None:
                    out[f"{PERF_CLASSES[cl]}.{name}"] = self._raw.param[cl * _PERF_MAX_PARAMS + j]
        return out


def param_names(lib=None):
    """{class name: [feature name or None per slot]} (imd_perf_param_name)."""
    h = load(lib) if not isinstance(lib, HostLibrary) else lib
    out = {}
    for cl, name in enumerate(PERF_CLASSES):
        names = []
        for j in range(_PERF_MAX_PARAMS):
            n = h.lib.imd_perf_param_name(cl, j)
            names.append(n.decode() if n is not None else None)
        out[name] = names
    return out


@dataclass(frozen=True)
class PhasePrediction:
    """imd_perf_phase_t: one phase's predicted time, its features and its §4 bound."""
    name: str        # the bench's phase tag ("K0 vec", "KL A", ...)
    cls: str         # the kernel class ("kl", "k4_128x256", ...)
    launches: int
    time_s: float
    bytes: float
    ops: float
    bound_s: float
    x: tuple         # features (the model: sum of param[cls][j] * x[j], ns; K4: max of
                     # the round-model terms j < 4 and the DRAM term j = 4)


@dataclass(frozen=True)
class Prediction:
    """imd_perf_t: the predicted call (status "SUCCESS", else the plan-time status and zeros)."""
    status: str
    dispatch: str
    k4_config: int
    sm_count: int
    time_s: float
    bound_s: float
    flops_eq: float
    tflops_eq: float
    kernel_launches: int
    workspace_bytes: int
    plan_device_bytes: int
    pack_bytes: int
    pack_ws_bytes: int
    phases: tuple    # PhasePrediction, in launch order (the call term last)

    def phase(self, name):
        """The PhasePrediction with this name, or None."""
        return next((p for p in self.phases if p.name == name), None)


def _calib_arg(calib):
    if calib is None:
        return None
    if isinstance(calib, Calibration):
        return C.byref(calib._raw)
    raise TypeError("calib must be a Calibration or None")


def _prediction(r, st):
    phases = tuple(PhasePrediction(PERF_PHASES[p.phase], PERF_CLASSES[p.cls], p.launches,
                                   p.time_s, p.bytes, p.ops, p.bound_s, tuple(p.x))
                   for p in r.phase[:r.num_phases])
    return Prediction(STATUS[st], DISPATCH[r.dispatch], r.k4_config, r.sm_count, r.time_s,
                      r.bound_s, r.flops_eq, r.tflops_eq, r.kernel_launches, r.workspace_bytes,
                      r.plan_device_bytes, r.pack_bytes, r.pack_ws_bytes, phases)


def predict(spec, extents=None, *, strides=None, N, g=0, rounding="RN", mode="EMULATE",
            vec_mode="GENERATE", packing=None, beta=False, alpha_zero=False, k4_config=None,
            calib=None, lib=None):
    """imd_perf_predict: the predicted imd_contract (DESIGN §6c). A plan-time status is
    returned in .status (with zeros); a bad argument raises. packing = (PackSpec or None,
    PackSpec or None): the packed plan. beta: C is read (beta != 0); alpha_zero: the beta-only
    call; k4_config: force an IMD_K4_* configuration (None: the plan-time rule)."""
    h, call = _prep(spec, extents, strides, N, g, rounding, mode, vec_mode, lib, packing)
    flags = (_PERF_BETA_NONZERO if beta else 0) | (_PERF_ALPHA_ZERO if alpha_zero else 0)
    req = _PerfRequest(C.sizeof(_PerfRequest), flags,
                       -1 if k4_config is None else int(k4_config), 0)
    r = _Perf()
    r.struct_size = C.sizeof(_Perf)
    pk = C.byref(call.packing) if packing is not None else None
    st = h.lib.imd_perf_predict(*call.args(), pk, C.byref(req), _calib_arg(calib), C.byref(r))
    return _prediction(r, st)


def predict_pack(pack, *, calib=None, lib=None):
    """imd_perf_predict_pack: the predicted imd_pack of a PackSpec."""
    h = load(lib) if not isinstance(lib, HostLibrary) else lib
    arg = _PackArg(pack)
    r = _Perf()
    r.struct_size = C.sizeof(_Perf)
    st = h.lib.imd_perf_predict_pack(C.byref(arg.d), _calib_arg(calib), C.byref(r))
    return _prediction(r, st)


@dataclass(frozen=True)
class Candidate:
    """imd_perf_candidate_t: one configuration, its predicted time and exact footprint."""
    num_moduli: int
    residue_group: int
    vec_mode: str
    packed: str          # "NONE", "A", "B", "AB" (the caller's operands)
    status: str
    feasible: bool
    k4_config: int
    dispatch: str
    time_s: float        # contract_s + pack_s / reuse: the rank key
    contract_s: float
    pack_s: float
    tflops_eq: float
    workspace_bytes: int
    plan_device_bytes: int
    pack_bytes: int
    pack_ws_bytes: int
    total_bytes: int
    kernel_launches: int


def rank(spec, extents=None, *, strides=None, N, g=None, vec_modes=("GENERATE",),
         packs=("NONE",), reuse=1, budget=0, beta=False, rounding="RN", mode="EMULATE",
         calib=None, lib=None):
    """imd_perf_rank: the candidates N x g x vec mode x packs ranked by predicted time inside a
    device-memory budget (bytes; 0 = none). N: an int or a list of admissible N; g: None
    (every g in 1..N) or a list; packs: any of "NONE", "A", "B", "AB"."""
    Ns = [int(N)] if isinstance(N, int) else [int(x) for x in N]
    gs = [] if g is None else ([int(g)] if isinstance(g, int) else [int(x) for x in g])
    h, call = _prep(spec, extents, strides, Ns[0], 0, rounding, mode, "GENERATE", lib)
    narr = (C.c_int32 * max(len(Ns), 1))(*Ns)
    garr = (C.c_int32 * max(len(gs), 1))(*gs)
    req = _RankRequest(C.sizeof(_RankRequest), C.sizeof(_Candidate), len(Ns), len(gs),
                       C.cast(narr, C.POINTER(C.c_int32)),
                       C.cast(garr, C.POINTER(C.c_int32)) if gs else None,
                       sum(_PERF_VEC[v] for v in set(vec_modes)),
                       sum(_PERF_PACK[p] for p in set(packs)), int(reuse),
                       _PERF_BETA_NONZERO if beta else 0, int(budget))
    count = C.c_int32(0)
    st = h.lib.imd_perf_rank(*call.args(), C.byref(req), _calib_arg(calib), None, 0,
                             C.byref(count))
    if st != 0:
        raise ImmerdinkError(st, "imd_perf_rank")
    out = (_Candidate * max(count.value, 1))()
    st = h.lib.imd_perf_rank(*call.args(), C.byref(req), _calib_arg(calib), out, count.value,
                             C.byref(count))
    if st != 0:
        raise ImmerdinkError(st, "imd_perf_rank")
    vec_name = {v: k for k, v in _VEC.items()}
    return [Candidate(c.num_moduli, c.residue_group, vec_name[c.vec_mode],
                      ["NONE", "A", "B", "AB"][c.packed], STATUS[c.status], bool(c.feasible),
                      c.k4_config, DISPATCH[c.dispatch], c.time_s, c.contract_s, c.pack_s,
                      c.tflops_eq, c.workspace_bytes, c.plan_device_bytes, c.pack_bytes,
                      c.pack_ws_bytes, c.total_bytes, c.kernel_launches)
            for c in out[:count.value]]


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
    ap.add_argument("--rank", default=None, metavar="N_LIST",
                    help="also rank configurations by predicted time (DESIGN §6c) for these N, "
                         "e.g. 8,14: every g, both vec modes, A / B packed or not")
    ap.add_argument("--budget", type=int, default=0, help="device-memory budget (bytes) for --rank")
    ap.add_argument("--reuse", type=int, default=1, help="contractions per pack for --rank")
    ap.add_argument("--calib", default=None, help="calibration file (default: the shipped one)")
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
    if a.rank:
        cal = Calibration.load(a.calib, a.lib) if a.calib else None
        cands = rank(a.spec, ext, N=[int(x) for x in a.rank.split(",")],
                     vec_modes=("GENERATE", "RESIDENT"), packs=("NONE", "A", "B", "AB"),
                     reuse=a.reuse, budget=a.budget, rounding=a.rounding, calib=cal, lib=a.lib)
        print(f"ranked by predicted time (reuse {a.reuse}, budget "
              f"{a.budget:,} B){'' if a.budget else ' = none'}; first 20 of {len(cands)}:")
        print(f"{'N':>3} {'g':>3} {'vec':>9} {'pack':>4} {'TFLOPS-eq':>9} {'time ms':>9} "
              f"{'contract':>9} {'pack ms':>8} {'total bytes':>15} {'fits':>4} status")
        for c in cands[:20]:
            print(f"{c.num_moduli:>3} {c.residue_group:>3} {c.vec_mode:>9} {c.packed:>4} "
                  f"{c.tflops_eq:>9.2f} {1e3 * c.time_s:>9.4f} {1e3 * c.contract_s:>9.4f} "
                  f"{1e3 * c.pack_s:>8.4f} {c.total_bytes:>15,} {'yes' if c.feasible else 'no':>4} "
                  f"{c.status}")


if __name__ == "__main__":
    _main()
