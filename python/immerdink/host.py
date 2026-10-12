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
import sys
from dataclasses import dataclass, replace

__all__ = ["ImmerdinkError", "HostLibrary", "load", "Desc", "contraction", "workspace_size",
           "workspace_layout", "plan_bytes", "lowering", "g_tradeoff", "PackSpec", "pack_layout",
           "pack_desc_for", "REGIONS", "STATUS", "PACK_MISMATCH", "Calibration", "predict",
           "predict_pack", "rank", "param_names", "PERF_CLASSES", "PERF_PHASES", "Accuracy",
           "ErrEstimate", "err_estimate", "min_moduli", "ERR_METRICS", "ERR_BASES",
           "ERR_STATUS", "INFEASIBLE", "PACK_FORMAT", "PACK_FORMAT_A16", "PACK_FORMAT_A64",
           "test_set_k_align", "CalibrationError", "CALIB_ERRORS", "CALIB_STRING_FIELDS"]

API_VERSION = 13  # the IMD_HOST_API_VERSION this wrapper was written for (minimum required)
# Pack formats (DESIGN v1.13 §4b, host API 9): the format of a pack is that of its k (2 iff
# k > 96); PACK_FORMAT is the newest format the library writes.
PACK_FORMAT_A16, PACK_FORMAT_A64 = 1, 2
PACK_FORMAT = PACK_FORMAT_A64

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
_PERF_PARAM_KL_DROP_SLOT = _PERF_PARAM_CAPACITY - 1  # IMD_PERF_PARAM_KL_DROP_SLOT (v1.19, 95)
_PERF_BETA_NONZERO, _PERF_ALPHA_ZERO = 1, 2
_PERF_VEC = {"GENERATE": 1, "RESIDENT": 2}
_PERF_PACK = {"NONE": 1, "A": 2, "B": 4, "AB": 8}
# v7 (DESIGN v1.10 §6c)
_PERF_KERNEL_CAPACITY, _PERF_KERNEL_COUNT, _PERF_CLASS_CAPACITY = 128, 90, 16
PERF_REGIMES = {"DEFAULT": 0, "BURST": 1, "SUSTAINED": 2}
PERF_CACHES = {"COLD": 0, "WARM": 1, "REPEAT": 2, "CLEAN": 3}  # v13 (DESIGN v1.18 §6c.8): CLEAN
PERF_ORDERS = {"TIME": 0, "TIES_BY_FOOTPRINT": 1}
# v8 (DESIGN v1.12 §6f): accuracy-aware planning
ERR_METRICS = {"NORMWISE": 0, "ELEMENTWISE": 1, "ABSOLUTE": 2}
ERR_BASES = {"BOUND": 0, "TYPICAL": 1}
ERR_STATUS = ["UNCHECKED", "MET", "MISSED"]
INFEASIBLE = {"STATUS": 1, "BUDGET": 2, "ACCURACY": 4}  # imd_perf_candidate_t.infeasible bits
# The parameters new in calibration format 2 (absent from a format-1 file); v12 (DESIGN v1.17
# §6c.1): kf.elem and pack.dram, optional (absent = 0).
FORMAT2_PARAMS = frozenset({"k0.chain", "kl.wave", "pack.chain", "pack.wave", "k5.chain",
                            "k5.dram",
                            "direct.chain", "beta.chain", "kf.elem", "pack.dram", "kl.chain"})


# Host API 10 (DESIGN §6c.3): why a calibration was rejected (IMD_PERF_CALIB_*, by value).
CALIB_ERRORS = ("OK", "ARGUMENT", "FILE", "SYNTAX", "UNKNOWN_KEY", "REPEATED_KEY", "VALUE",
                "TOO_LONG", "MISSING_KEY", "FORMAT2_KEY", "INCONSISTENT")


class CalibrationError(ValueError):
    """A calibration the C parser rejected (host API 10): .line (1-based; 0: not tied to a
    line), .reason (a CALIB_ERRORS name, e.g. "TOO_LONG"), .key (the offending or missing key,
    or "") and the library's message."""

    def __init__(self, line, reason, key, message, what=""):
        self.line, self.key, self.message = int(line), key, message
        self.reason = CALIB_ERRORS[reason] if 0 <= reason < len(CALIB_ERRORS) else f"<{reason}>"
        super().__init__(f"{what}{': ' if what else ''}invalid calibration "
                         f"[{self.reason}] {message}")


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
                ("packed", C.c_int32), ("pack_mismatch", C.c_int32),
                ("k_align", C.c_int32), ("reserved6", C.c_int32)]   # v6 (DESIGN v1.13 §4b)


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
                ("param", C.c_double * _PERF_PARAM_CAPACITY),
                # v7 (format 2)
                ("kernel", (C.c_int32 * 4) * _PERF_KERNEL_CAPACITY),
                ("clock_eff_mhz", C.c_double), ("clock_min_mhz", C.c_double),
                ("power_cap_w", C.c_double), ("power_idle_w", C.c_double),
                ("energy_tc_pj", C.c_double), ("energy_alu_pj", C.c_double),
                ("energy_byte_pj", C.c_double), ("energy_gamma", C.c_double),
                ("warm_l2_bytes", C.c_double), ("warm_l2byte_ns", C.c_double),
                ("band_call", (C.c_double * 2) * 3),
                ("band_class", (C.c_double * 2) * _PERF_CLASS_CAPACITY),
                ("tie_ratio", C.c_double),
                # v11 (DESIGN v1.15 §6c.7)
                ("energy_byte_f_pj", C.c_double),
                # v12 (DESIGN v1.17 §6c.1)
                ("pack_l2_hit_bytes", C.c_double),
                # v13 (DESIGN v1.18 §6c.8)
                ("clock_boost_mhz", C.c_double), ("power_act_w", C.c_double),
                ("power_act_gap_us", C.c_double), ("energy_dram_pj", C.c_double),
                ("warm_call_ns", C.c_double), ("warm_clean_byte_ns", C.c_double),
                ("pack_l2_planes", C.c_double)]


# The string fields of a calibration and their capacities in bytes, NUL included (the
# generation-time check of bench/perf_fit.py reads them from here).
CALIB_STRING_FIELDS = {n: getattr(_PerfCalib, n).size for n in ("gpu", "date", "version",
                                                                  "clocks", "notes")}


class _PerfCalibError(C.Structure):  # host API 10 (DESIGN §6c.3)
    _fields_ = [("struct_size", C.c_uint32), ("line", C.c_int32), ("reason", C.c_int32),
                ("reserved", C.c_int32), ("key", C.c_char * 64), ("message", C.c_char * 192)]


class _ErrRequest(C.Structure):  # v8 (DESIGN §6f)
    _fields_ = [("struct_size", C.c_uint32), ("metric", C.c_int32), ("target", C.c_double),
                ("cond", C.c_double), ("norm_product", C.c_double),
                ("l1_ratio_a", C.c_double), ("l1_ratio_b", C.c_double),
                ("basis", C.c_int32), ("reserved", C.c_int32), ("fail_prob", C.c_double)]


class _Err(C.Structure):  # v8 (DESIGN §6f)
    _fields_ = [("struct_size", C.c_uint32), ("status", C.c_int32), ("num_moduli", C.c_int32),
                ("rounding", C.c_int32), ("dispatch", C.c_int32), ("err_status", C.c_int32),
                ("k", C.c_int64), ("outputs", C.c_int64), ("bound", C.c_double),
                ("typical", C.c_double), ("unit_bound", C.c_double),
                ("unit_typical", C.c_double)]


class _PerfRequest(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("flags", C.c_int32), ("k4_config", C.c_int32),
                ("reserved", C.c_int32), ("eps_fraction", C.c_double),
                ("regime", C.c_int32), ("cache", C.c_int32), ("clock_mhz", C.c_double),  # v7
                ("accuracy", C.POINTER(_ErrRequest))]  # v8


class _PerfPhase(C.Structure):
    _fields_ = [("phase", C.c_int32), ("cls", C.c_int32), ("launches", C.c_int64),
                ("time_s", C.c_double), ("bytes", C.c_double), ("ops", C.c_double),
                ("bound_s", C.c_double), ("x", C.c_double * _PERF_MAX_PARAMS)]


class _Perf(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("status", C.c_int32), ("dispatch", C.c_int32),
                ("k4_config", C.c_int32), ("sm_count", C.c_int32), ("num_phases", C.c_int32),
                ("time_s", C.c_double), ("bound_s", C.c_double), ("flops_eq", C.c_double),
                ("tflops_eq", C.c_double), ("eps_s", C.c_double), ("eps_fraction", C.c_double),
                ("kernel_launches", C.c_int64),
                ("workspace_bytes", C.c_uint64), ("plan_device_bytes", C.c_uint64),
                ("pack_bytes", C.c_uint64), ("pack_ws_bytes", C.c_uint64),
                ("phase", _PerfPhase * _PERF_MAX_PHASES),
                # v7
                ("regime", C.c_int32), ("cache", C.c_int32), ("clock_mhz", C.c_double),
                ("power_w", C.c_double), ("warm_bytes", C.c_double),
                ("time_lo_s", C.c_double), ("time_hi_s", C.c_double),
                ("phase_lo_s", C.c_double * _PERF_MAX_PHASES),
                ("phase_hi_s", C.c_double * _PERF_MAX_PHASES),
                # v8
                ("err_bound", C.c_double), ("err_typical", C.c_double),
                ("err_status", C.c_int32), ("err_metric", C.c_int32)]


class _RankRequest(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("candidate_size", C.c_uint32),
                ("num_moduli_count", C.c_int32), ("residue_group_count", C.c_int32),
                ("num_moduli", C.POINTER(C.c_int32)), ("residue_groups", C.POINTER(C.c_int32)),
                ("vec_modes", C.c_int32), ("pack_modes", C.c_int32), ("reuse", C.c_int32),
                ("flags", C.c_int32), ("budget_bytes", C.c_uint64),
                ("eps_fraction", C.c_double),
                ("regime", C.c_int32), ("cache", C.c_int32), ("order", C.c_int32),
                ("reserved", C.c_int32), ("clock_mhz", C.c_double),  # v7
                ("accuracy", C.POINTER(_ErrRequest))]  # v8


class _Candidate(C.Structure):
    _fields_ = [("num_moduli", C.c_int32), ("residue_group", C.c_int32), ("vec_mode", C.c_int32),
                ("packed", C.c_int32), ("status", C.c_int32), ("feasible", C.c_int32),
                ("k4_config", C.c_int32), ("dispatch", C.c_int32), ("time_s", C.c_double),
                ("contract_s", C.c_double), ("pack_s", C.c_double), ("tflops_eq", C.c_double),
                ("eps_s", C.c_double), ("eps_fraction", C.c_double),
                ("workspace_bytes", C.c_uint64), ("plan_device_bytes", C.c_uint64),
                ("pack_bytes", C.c_uint64), ("pack_ws_bytes", C.c_uint64),
                ("total_bytes", C.c_uint64), ("kernel_launches", C.c_int64),
                ("time_lo_s", C.c_double), ("time_hi_s", C.c_double),  # v7
                ("clock_mhz", C.c_double), ("power_w", C.c_double),
                ("tie", C.c_int32), ("raw_rank", C.c_int32),
                ("err_bound", C.c_double), ("err_typical", C.c_double),  # v8
                ("err_status", C.c_int32), ("infeasible", C.c_int32)]


assert C.sizeof(_Lowering) == 192  # version 6 (DESIGN v1.13 §4b; v5 = 184)
assert _Lowering.k_align.offset == 184
assert C.sizeof(_Options) == 24
assert C.sizeof(_PackDesc) == 72 and C.sizeof(_PackLayout) == 136
assert C.sizeof(_PerfCalib) == 3744 and _PerfCalib.energy_byte_f_pj.offset == 3672 and \
    _PerfCalib.pack_l2_hit_bytes.offset == 3680 and _PerfCalib.clock_boost_mhz.offset == 3688 and \
    _PerfCalib.pack_l2_planes.offset == 3736 and C.sizeof(_PerfPhase) == 96 and C.sizeof(_Perf) == 1976
assert C.sizeof(_RankRequest) == 96 and C.sizeof(_Candidate) == 192
assert C.sizeof(_PerfRequest) == 48
assert C.sizeof(_ErrRequest) == 64 and C.sizeof(_Err) == 72


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
            # v10 (DESIGN §6c.3): calibration parse errors with the reason
            "imd_perf_calib_parse_ex": (C.c_int, [C.c_char_p, C.c_size_t, C.POINTER(_PerfCalib),
                                                  C.POINTER(_PerfCalibError)]),
            "imd_perf_calib_load_ex": (C.c_int, [C.c_char_p, C.POINTER(_PerfCalib),
                                                 C.POINTER(_PerfCalibError)]),
            "imd_perf_calib_default_ex": (C.c_int, [C.POINTER(_PerfCalib),
                                                    C.POINTER(_PerfCalibError)]),
            "imd_perf_calib_error_name": (C.c_char_p, [C.c_int32]),
            "imd_perf_class_name": (C.c_char_p, [C.c_int32]),
            "imd_perf_param_name": (C.c_char_p, [C.c_int32, C.c_int32]),
            "imd_perf_phase_name": (C.c_char_p, [C.c_int32]),
            "imd_perf_kernel_name": (C.c_char_p, [C.c_int32]),  # v7
            "imd_perf_predict": (C.c_int, [T, T, T, O, C.POINTER(_Packing),
                                           C.POINTER(_PerfRequest), C.POINTER(_PerfCalib),
                                           C.POINTER(_Perf)]),
            "imd_perf_predict_pack": (C.c_int, [C.POINTER(_PackDesc), C.POINTER(_PerfRequest),
                                                C.POINTER(_PerfCalib), C.POINTER(_Perf)]),
            "imd_perf_rank": (C.c_int, [T, T, T, O, C.POINTER(_RankRequest),
                                        C.POINTER(_PerfCalib), C.POINTER(_Candidate), C.c_int32,
                                        C.POINTER(C.c_int32)]),
            # v8 (DESIGN §6f): accuracy-aware planning
            "imd_err_estimate": (C.c_int, [C.c_int32, C.c_int32, C.c_int32, C.c_int64,
                                           C.c_int64, C.POINTER(_ErrRequest), C.POINTER(_Err)]),
            "imd_err_min_moduli": (C.c_int, [T, T, T, O, C.c_int32, C.c_int32,
                                             C.POINTER(_ErrRequest), C.POINTER(_Err)]),
        }
        for name, (res, args) in sig.items():
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
        self.version = lib.imd_host_api_version()
        if self.version < API_VERSION:
            raise OSError(f"{path}: host API version {self.version} < {API_VERSION}")
        # Host API 10 (DESIGN §6c.3): whether the shipped calibration parses, checked once.
        # If not, every query with calib=None (the C library's NULL) would only say
        # INVALID_VALUE: predict / predict_pack / rank / planner.Query raise this instead.
        c, e = _PerfCalib(), _PerfCalibError(C.sizeof(_PerfCalibError))
        c.struct_size = C.sizeof(_PerfCalib)
        st = lib.imd_perf_calib_default_ex(C.byref(c), C.byref(e))
        self.shipped_calib_error = None if st == 0 else CalibrationError(
            e.line, e.reason, e.key.decode(errors="replace"),
            e.message.decode(errors="replace"), path)

    def check_shipped_calib(self):
        """Raise the shipped calibration's CalibrationError, if it has one."""
        if self.shipped_calib_error is not None:
            raise self.shipped_calib_error


_loaded = {}  # path (None: the default search) -> HostLibrary


def test_set_k_align(a, *, lib=None):
    """TEST ONLY (DESIGN v1.13 §4b.5): libimmerdink_host's imd_test_set_k_align. a = 0
    restores AUTO (A = 64 iff k > 96); 16 or 64 forces the K alignment of every sizing query,
    lowering, plan and pack created afterwards in this process (device plans follow).
    ValueError for another value. The plan cache key does not include the hook, so this also
    clears immerdink.device's default PlanCache (when that module is loaded); tests that set
    the hook use fresh caches (PlanCache, planner.Memo) and restore AUTO afterwards."""
    h = load(lib) if not isinstance(lib, HostLibrary) else lib
    f = h.lib.imd_test_set_k_align
    f.restype, f.argtypes = C.c_int, [C.c_int32]
    if f(int(a)) != 0:
        raise ValueError(f"k alignment {a!r}: 0 (AUTO), 16 or 64")
    dev = sys.modules.get(__package__ + ".device") if __package__ else None
    if dev is not None:
        dev.default_plan_cache.clear()


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

    def with_moduli(self, N):
        """This descriptor at another N (DESIGN v1.19 §6g, a precision phase): every other
        field, the orders included, unchanged. N is part of a pack's key, so a pack serves
        one N; a buffer of pack_bytes at the largest N holds the pack at any smaller N (§6a
        C-pack)."""
        return replace(self, num_moduli=int(N))


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
    k_align: int           # v6: the plan's K alignment A (16 / 64), k_hat = ceil(k)_A
    reserved6: int


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

    @staticmethod
    def _raise(st, e, what):
        """CalibrationError from a failed _ex call (an INVALID_VALUE with a reason), else
        ImmerdinkError (allocation failure)."""
        if st == 1 and e.reason != 0:
            raise CalibrationError(e.line, e.reason, e.key.decode(errors="replace"),
                                   e.message.decode(errors="replace"), what)
        raise ImmerdinkError(st, what)

    @classmethod
    def default(cls, lib=None):
        """The shipped calibration (imd_perf_calib_default_ex); CalibrationError with the
        embedded file's own parse error if it does not parse (DESIGN §6c.3)."""
        h = load(lib) if not isinstance(lib, HostLibrary) else lib
        c, e = cls._new(), _PerfCalibError(C.sizeof(_PerfCalibError))
        st = h.lib.imd_perf_calib_default_ex(C.byref(c), C.byref(e))
        if st != 0:
            cls._raise(st, e, "imd_perf_calib_default")
        return cls(c, h)

    @classmethod
    def parse(cls, text, lib=None):
        """Parse calibration text (format 1 or 2); CalibrationError (a ValueError: line,
        reason, key, message) if it is invalid."""
        h = load(lib) if not isinstance(lib, HostLibrary) else lib
        data = text.encode() if isinstance(text, str) else bytes(text)
        c, e = cls._new(), _PerfCalibError(C.sizeof(_PerfCalibError))
        st = h.lib.imd_perf_calib_parse_ex(data, len(data), C.byref(c), C.byref(e))
        if st != 0:
            cls._raise(st, e, "")
        return cls(c, h)

    @classmethod
    def load(cls, path, lib=None):
        """Load a calibration file; CalibrationError (a ValueError) if it is unreadable or
        invalid."""
        h = load(lib) if not isinstance(lib, HostLibrary) else lib
        c, e = cls._new(), _PerfCalibError(C.sizeof(_PerfCalibError))
        st = h.lib.imd_perf_calib_load_ex(os.fsencode(path), C.byref(c), C.byref(e))
        if st != 0:
            cls._raise(st, e, os.fsdecode(path))
        return cls(c, h)

    @classmethod
    def from_values(cls, meta, params, lib=None, extra=None):
        """A calibration from a metadata dict and a {"<class>.<feature>": value} dict, through
        the calibration text (so the C parser validates it). extra (v7): {key: value or tuple}
        of the format-2 keys ("kernel.kf16": (2, 128, 48640, 256), "power.cap_w": 450,
        "band.call_small": (0.9, 1.1), ...); with extra the text is format 2 (every format-2
        key is then required), else format 1."""
        keys = ("gpu", "sm_count", "date", "version", "clocks", "notes",
                "peak_dram_bytes_per_s", "peak_int8_ops_per_s", "peak_fp64_flops_per_s",
                "l2_bytes")
        lines = [f"format = {2 if extra else 1}"] + [f"{k} = {meta[k]}" for k in keys if k in meta]
        lines += [f"{k} = {float(v)!r}" for k, v in params.items()]
        for k, v in (extra or {}).items():
            vals = v if isinstance(v, (tuple, list)) else (v,)
            lines.append(f"{k} = " + " ".join(repr(x) if isinstance(x, float) else str(x)
                                              for x in vals))
        return cls.parse("\n".join(lines) + "\n", lib)

    @property
    def file_format(self):
        """v7: the calibration format (1 or 2)."""
        return self._raw.format

    @property
    def extra(self):
        """v7: the format-2 keys as from_values takes them ({} for a format-1 calibration)."""
        r = self._raw
        if r.format != 2:
            return {}
        out = {f"kernel.{n}": tuple(r.kernel[k]) for k, n in enumerate(kernel_names(self._lib))}
        out.update(self.power)
        out.update(self.warm)
        out.update(self.bands)
        out["rank.tie_ratio"] = r.tie_ratio
        return out

    @property
    def kernels(self):
        """v7: {kernel name: (CTAs per SM, registers, shared memory bytes, threads)}."""
        r = self._raw
        return {n: tuple(r.kernel[k]) for k, n in enumerate(kernel_names(self._lib))}

    @property
    def power(self):
        r = self._raw
        return {"power.clock_eff_mhz": r.clock_eff_mhz, "power.clock_min_mhz": r.clock_min_mhz,
                "power.cap_w": r.power_cap_w, "power.idle_w": r.power_idle_w,
                "power.tc_pj": r.energy_tc_pj, "power.alu_pj": r.energy_alu_pj,
                "power.byte_pj": r.energy_byte_pj, "power.gamma": r.energy_gamma,
                "power.byte_f_pj": r.energy_byte_f_pj,  # v11 (§6c.7; 0: absent)
                # v13 (DESIGN v1.18 §6c.8; 0: absent)
                "power.clock_boost_mhz": r.clock_boost_mhz, "power.act_w": r.power_act_w,
                "power.act_gap_us": r.power_act_gap_us, "power.dram_pj": r.energy_dram_pj}

    @property
    def warm(self):
        r = self._raw
        return {"warm.l2_bytes": r.warm_l2_bytes, "warm.l2byte": r.warm_l2byte_ns,
                "warm.pack_l2_bytes": r.pack_l2_hit_bytes,  # v12 (§6c.1; 0: absent)
                # v13 (DESIGN v1.18 §6c.8; 0: absent)
                "warm.pack_l2_planes": r.pack_l2_planes, "warm.call_ns": r.warm_call_ns,
                "warm.clean_byte": r.warm_clean_byte_ns,
                # v1.19 (§6c.9; 0: absent): stored in param[IMD_PERF_PARAM_KL_DROP_SLOT]
                "warm.kl_drop": r.param[_PERF_PARAM_KL_DROP_SLOT]}

    @property
    def bands(self):
        r = self._raw
        out = {f"band.call_{n}": tuple(r.band_call[i]) for i, n in
               enumerate(("small", "mid", "large"))}
        out.update({f"band.{c}": tuple(r.band_class[i]) for i, c in enumerate(PERF_CLASSES)})
        return out

    def format(self):
        """The canonical text in the calibration's own format, 1 or 2 (imd_perf_calib_format)."""
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
                key = f"{PERF_CLASSES[cl]}.{name}"
                if name is not None and (self._raw.format == 2 or key not in FORMAT2_PARAMS):
                    out[key] = self._raw.param[cl * _PERF_MAX_PARAMS + j]
        return out


def kernel_names(lib=None):
    """v7: the kernel names of the occupancy table (imd_perf_kernel_name), in index order."""
    h = load(lib) if not isinstance(lib, HostLibrary) else lib
    return [h.lib.imd_perf_kernel_name(k).decode() for k in range(_PERF_KERNEL_COUNT)]


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


# ------------------------------------------------------------------------------------------
# Accuracy-aware planning (DESIGN v1.12 §6f): marshalling only; the bounds are the C library's

@dataclass(frozen=True)
class Accuracy:
    """imd_err_request_t: the error metric, the target and the caller's operand statistics.

    Per output x_ij = sum_h a_ih b_hj of the GEMM view (a_i: A's fiber over the contracted
    labels, b_j: B's), D the emulated product:
      metric "NORMWISE" |D - x| / (||a_i||_2 ||b_j||_2) (needs nothing), "ELEMENTWISE"
        |D - x| / |x| (needs cond = rho >= max ||a_i|| ||b_j|| / |x_ij|), "ABSOLUTE" |D - x|
        (needs norm_product >= max ||a_i|| ||b_j||);
      target > 0: rank / min_moduli keep the configurations whose estimate (basis "BOUND":
        the rigorous bound, the default; "TYPICAL": the typical estimate, not a guarantee)
        is <= target; 0: estimates reported only. TYPICAL assumes quantization errors that
        behave randomly; it fails (up to ~sqrt(k) / sqrt(2 ln(2M/p)) too low) on rows of
        many equal values not representable at the row's scale, e.g. uniform superpositions
        (entries 1/sqrt(n)) or repeated hopping constants: use "BOUND" for such operands;
      l1_ratio_a / l1_ratio_b: optional upper bounds (>= 1) on max ||a_i||_1 / ||a_i||_2
        (tighter bounds for decaying or sparse rows; 0 = unknown);
      fail_prob: the typical estimate's failure probability per call (0: 2^-40; for a probability P over C calls of a run pass P / C).
    The bounds are on the product D (alpha = 1, beta = 0). For C = beta C + RN(alpha D):
    alpha != 1 turns a bound n into n (1 + u) + u (u = 2^-53; ABSOLUTE: pass |alpha| times
    the norm_product); beta != 0 adds u |beta C + alpha D|, which these statistics cannot
    express. Immutable and hashable; the C struct is built once (planner.py passes it by
    pointer)."""
    metric: str = "NORMWISE"
    target: float = 0.0
    cond: float = 0.0
    norm_product: float = 0.0
    l1_ratio_a: float = 0.0
    l1_ratio_b: float = 0.0
    basis: str = "BOUND"
    fail_prob: float = 0.0

    def __post_init__(self):
        if self.metric not in ERR_METRICS:
            raise ValueError(f"metric must be one of {sorted(ERR_METRICS)}, got {self.metric!r}")
        if self.basis not in ERR_BASES:
            raise ValueError(f"basis must be one of {sorted(ERR_BASES)}, got {self.basis!r}")
        c = _ErrRequest(C.sizeof(_ErrRequest), ERR_METRICS[self.metric], float(self.target),
                        float(self.cond), float(self.norm_product), float(self.l1_ratio_a),
                        float(self.l1_ratio_b), ERR_BASES[self.basis], 0, float(self.fail_prob))
        object.__setattr__(self, "_c", c)                 # not a dataclass field: no eq / hash
        object.__setattr__(self, "_ptr", C.pointer(c))


def _accuracy_ptr(accuracy):
    if accuracy is None:
        return None
    if isinstance(accuracy, Accuracy):
        return accuracy._ptr
    raise TypeError("accuracy must be a host.Accuracy or None")


@dataclass(frozen=True)
class ErrEstimate:
    """imd_err_t: the error estimates of one configuration (DESIGN §6f.2), in the request's
    metric. bound: rigorous (inf: none); typical: probabilistic (RN EMULATE; else = bound);
    unit_*: the coefficient of ||a_i|| ||b_j|| before the metric. min_moduli: num_moduli is
    the answer (0 and status "NOT_SUPPORTED": no N in the range meets the target)."""
    status: str
    num_moduli: int
    rounding: str
    dispatch: str
    err_status: str      # "UNCHECKED" (no target), "MET", "MISSED"
    k: int
    outputs: int
    bound: float
    typical: float
    unit_bound: float
    unit_typical: float


_ROUNDING_NAME = {v: k for k, v in _ROUNDING.items()}


def _err_estimate(r, st):
    return ErrEstimate(STATUS[st], r.num_moduli, _ROUNDING_NAME.get(r.rounding, "RN"),
                       DISPATCH[r.dispatch], ERR_STATUS[r.err_status], r.k, r.outputs, r.bound,
                       r.typical, r.unit_bound, r.unit_typical)


def err_estimate(N, k, *, rounding="RN", dispatch="EMULATE", outputs=1, accuracy=None,
                 lib=None):
    """imd_err_estimate: the error estimates of N moduli at contracted length k with `outputs`
    outputs (the typical estimate's union bound), from scalars; a bad argument raises."""
    h = load(lib) if not isinstance(lib, HostLibrary) else lib
    r = _Err()
    r.struct_size = C.sizeof(_Err)
    st = h.lib.imd_err_estimate(int(N), _ROUNDING[rounding], DISPATCH.index(dispatch), int(k),
                                int(outputs), _accuracy_ptr(accuracy), C.byref(r))
    if st != 0:
        raise ImmerdinkError(st, "imd_err_estimate")
    return _err_estimate(r, st)


def min_moduli(spec, extents=None, *, accuracy, strides=None, rounding="RN", mode="EMULATE",
               n_lo=0, n_hi=0, lib=None):
    """imd_err_min_moduli: the smallest N in [n_lo, n_hi] (0, 0: 2..20) whose estimate meets
    accuracy.target for this contraction (status "SUCCESS"), else num_moduli 0 and status
    "NOT_SUPPORTED" with err_status "MISSED" (unattainable in the range). A plan-time status
    of the lowering (e.g. "NOT_SUPPORTED" for a repeated C label: err_status "UNCHECKED") or
    an argument error ("INVALID_VALUE", e.g. no target) is returned in .status with zeros;
    nothing raises (as predict)."""
    h, call = _prep(spec, extents, strides, 20, 1, rounding, mode, "GENERATE", lib)
    r = _Err()
    r.struct_size = C.sizeof(_Err)
    st = h.lib.imd_err_min_moduli(*call.args(), int(n_lo), int(n_hi), _accuracy_ptr(accuracy),
                                  C.byref(r))
    return _err_estimate(r, st)


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
    lo_s: float = 0.0  # v7: the phase's band
    hi_s: float = 0.0


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
    eps_s: float          # 0 since host API 11 (DESIGN v1.15 §6c.1: the eps path is retired)
    eps_fraction: float   # 0 since host API 11
    kernel_launches: int
    workspace_bytes: int
    plan_device_bytes: int
    pack_bytes: int
    pack_ws_bytes: int
    phases: tuple    # PhasePrediction, in launch order (the call term last)
    regime: str = "BURST"     # v7: the regime used, the cache state, the SM clock (0: no
    cache: str = "COLD"       # clock model), the call's power there (0: no power model), the
    clock_mhz: float = 0.0    # input bytes served from L2, and the band of time_s
    power_w: float = 0.0
    warm_bytes: float = 0.0
    time_lo_s: float = 0.0
    time_hi_s: float = 0.0
    err_bound: float = 0.0        # v8 (DESIGN §6f): rigorous error bound and typical estimate
    err_typical: float = 0.0      # in the request's metric (NORMWISE without accuracy=),
    err_status: str = "UNCHECKED"  # "MET" / "MISSED" with a target; imd_perf_predict_pack: 0
    err_metric: str = "NORMWISE"

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
                                   p.time_s, p.bytes, p.ops, p.bound_s, tuple(p.x),
                                   r.phase_lo_s[i], r.phase_hi_s[i])
                   for i, p in enumerate(r.phase[:r.num_phases]))
    reg = {v: k for k, v in PERF_REGIMES.items()}
    cac = {v: k for k, v in PERF_CACHES.items()}
    return Prediction(STATUS[st], DISPATCH[r.dispatch], r.k4_config, r.sm_count, r.time_s,
                      r.bound_s, r.flops_eq, r.tflops_eq, r.eps_s, r.eps_fraction,
                      r.kernel_launches, r.workspace_bytes, r.plan_device_bytes, r.pack_bytes,
                      r.pack_ws_bytes, phases, reg.get(r.regime, "BURST"),
                      cac.get(r.cache, "COLD"), r.clock_mhz, r.power_w, r.warm_bytes,
                      r.time_lo_s, r.time_hi_s, r.err_bound, r.err_typical,
                      ERR_STATUS[r.err_status], _ERR_METRIC_NAME[r.err_metric])


_ERR_METRIC_NAME = {v: k for k, v in ERR_METRICS.items()}


def _eps(eps_fraction):
    return -1.0 if eps_fraction is None else float(eps_fraction)


def predict(spec, extents=None, *, strides=None, N, g=0, rounding="RN", mode="EMULATE",
            vec_mode="GENERATE", packing=None, beta=False, alpha_zero=False, k4_config=None,
            eps_fraction=None, regime="DEFAULT", cache="COLD", clock_mhz=0.0, accuracy=None,
            calib=None, lib=None):
    """imd_perf_predict: the predicted imd_contract (DESIGN §6c). A plan-time status is
    returned in .status (with zeros); a bad argument raises. packing = (PackSpec or None,
    PackSpec or None): the packed plan. beta: C is read (beta != 0); alpha_zero: the beta-only
    call; k4_config: force an IMD_K4_* configuration (None: the plan-time rule); eps_fraction:
    validated (None, < 0 or 0..1) and ignored since host API 11 (the eps path is retired,
    DESIGN v1.15 §6c.1). v7: regime "DEFAULT" / "BURST" / "SUSTAINED", cache "COLD" / "WARM" /
    "REPEAT" (v13: "CLEAN", an L2 flushed by reads), clock_mhz > 0: predict at that SM clock. v8: accuracy (an Accuracy): err_bound /
    err_typical in its metric and err_status against its target (None: NORMWISE, no target)."""
    h, call = _prep(spec, extents, strides, N, g, rounding, mode, vec_mode, lib, packing)
    flags = (_PERF_BETA_NONZERO if beta else 0) | (_PERF_ALPHA_ZERO if alpha_zero else 0)
    req = _PerfRequest(C.sizeof(_PerfRequest), flags,
                       -1 if k4_config is None else int(k4_config), 0, _eps(eps_fraction),
                       PERF_REGIMES[regime], PERF_CACHES[cache], float(clock_mhz),
                       _accuracy_ptr(accuracy))
    r = _Perf()
    r.struct_size = C.sizeof(_Perf)
    pk = C.byref(call.packing) if packing is not None else None
    if calib is None:
        h.check_shipped_calib()
    st = h.lib.imd_perf_predict(*call.args(), pk, C.byref(req), _calib_arg(calib), C.byref(r))
    return _prediction(r, st)


def predict_pack(pack, *, eps_fraction=None, regime="DEFAULT", cache="COLD", clock_mhz=0.0,
                 calib=None, lib=None):
    """imd_perf_predict_pack: the predicted imd_pack of a PackSpec."""
    h = load(lib) if not isinstance(lib, HostLibrary) else lib
    arg = _PackArg(pack)
    r = _Perf()
    r.struct_size = C.sizeof(_Perf)
    req = _PerfRequest(C.sizeof(_PerfRequest), 0, -1, 0, _eps(eps_fraction),
                       PERF_REGIMES[regime], PERF_CACHES[cache], float(clock_mhz))
    if calib is None:
        h.check_shipped_calib()
    st = h.lib.imd_perf_predict_pack(C.byref(arg.d), C.byref(req), _calib_arg(calib), C.byref(r))
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
    eps_s: float         # 0 since host API 11 (eps path retired)
    eps_fraction: float  # 0 since host API 11
    workspace_bytes: int
    plan_device_bytes: int
    pack_bytes: int
    pack_ws_bytes: int
    total_bytes: int
    kernel_launches: int
    time_lo_s: float = 0.0   # v7: band of time_s, the contract's clock and power, tie with
    time_hi_s: float = 0.0   # the best (ORDER_BAND) and the place in the v6 time order
    clock_mhz: float = 0.0
    power_w: float = 0.0
    tie: bool = False
    raw_rank: int = 0
    err_bound: float = 0.0   # v8 (DESIGN §6f): the error estimates in the request's metric
    err_typical: float = 0.0  # (NORMWISE without accuracy=; inf when status != SUCCESS),
    err_status: str = "UNCHECKED"  # against its target, and why it is infeasible
    infeasible: int = 0      # INFEASIBLE bits: STATUS 1, BUDGET 2, ACCURACY 4


def rank(spec, extents=None, *, strides=None, N, g=None, vec_modes=("GENERATE",),
         packs=("NONE",), reuse=1, budget=0, beta=False, rounding="RN", mode="EMULATE",
         eps_fraction=None, order="TIME", regime="DEFAULT", cache="COLD", clock_mhz=0.0,
         accuracy=None, calib=None, lib=None):
    """imd_perf_rank: the candidates N x g x vec mode x packs ranked by predicted time inside a
    device-memory budget (bytes; 0 = none). N: an int or a list of admissible N; g: None
    (every g in 1..N) or a list; packs: any of "NONE", "A", "B", "AB". v7: order "TIME" (the
    default) or "TIES_BY_FOOTPRINT" (the candidates within the calibration's tie ratio of the
    best first, by footprint; Candidate.tie is set in either order), regime / cache / clock_mhz
    as predict. v8: accuracy (an Accuracy with a target): the candidates whose estimate misses
    the target are infeasible (infeasible bit ACCURACY), so the first candidate, if feasible,
    is the cheapest configuration that meets the target."""
    Ns = [int(N)] if hasattr(N, "__index__") else [int(x) for x in N]
    gs = [] if g is None else ([int(g)] if isinstance(g, int) else [int(x) for x in g])
    h, call = _prep(spec, extents, strides, Ns[0], 0, rounding, mode, "GENERATE", lib)
    narr = (C.c_int32 * max(len(Ns), 1))(*Ns)
    garr = (C.c_int32 * max(len(gs), 1))(*gs)
    req = _RankRequest(C.sizeof(_RankRequest), C.sizeof(_Candidate), len(Ns), len(gs),
                       C.cast(narr, C.POINTER(C.c_int32)),
                       C.cast(garr, C.POINTER(C.c_int32)) if gs else None,
                       sum(_PERF_VEC[v] for v in set(vec_modes)),
                       sum(_PERF_PACK[p] for p in set(packs)), int(reuse),
                       _PERF_BETA_NONZERO if beta else 0, int(budget), _eps(eps_fraction),
                       PERF_REGIMES[regime], PERF_CACHES[cache], PERF_ORDERS[order], 0,
                       float(clock_mhz), _accuracy_ptr(accuracy))
    count = C.c_int32(0)
    if calib is None:
        h.check_shipped_calib()
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
                      c.tflops_eq, c.eps_s, c.eps_fraction, c.workspace_bytes,
                      c.plan_device_bytes, c.pack_bytes,
                      c.pack_ws_bytes, c.total_bytes, c.kernel_launches, c.time_lo_s,
                      c.time_hi_s, c.clock_mhz, c.power_w, bool(c.tie), c.raw_rank,
                      c.err_bound, c.err_typical, ERR_STATUS[c.err_status], c.infeasible)
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
