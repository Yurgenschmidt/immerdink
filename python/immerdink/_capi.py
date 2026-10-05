"""ctypes declarations of include/immerdink/immerdink.h (DESIGN §6, §6b) and the loading of
libimmerdink. No logic: the device API (immerdink.device) builds on this.

The library: `path`, else $IMMERDINK_LIB (the .so), else $IMMERDINK_LIB_DIR/libimmerdink.so,
else $IMMERDINK_BUILD_DIR or <repo>/build, else the system search path (as host.py)."""

import ctypes as C
import ctypes.util
import os
import threading

from . import host

API_VERSION = 5  # IMD_HOST_API_VERSION needed: packed operands (§6b)


class Tensor(C.Structure):          # imd_tensor_t
    _fields_ = [("rank", C.c_int32), ("extent", C.POINTER(C.c_int64)),
                ("stride", C.POINTER(C.c_int64)), ("label", C.POINTER(C.c_int32))]


class Options(C.Structure):         # imd_options_t
    _fields_ = [("num_moduli", C.c_int32), ("residue_group", C.c_int32),
                ("rounding", C.c_int32), ("mode", C.c_int32), ("vec_mode", C.c_int32),
                ("k4_tune", C.c_int32)]


class PackDesc(C.Structure):        # imd_pack_desc_t
    _fields_ = [("struct_size", C.c_uint32), ("num_moduli", C.c_int32), ("rounding", C.c_int32),
                ("order", C.c_int32), ("tensor", C.POINTER(Tensor)),
                ("num_contracted", C.c_int32), ("contracted", C.POINTER(C.c_int32)),
                ("num_batch", C.c_int32), ("batch", C.POINTER(C.c_int32)),
                ("num_rows", C.c_int32), ("rows", C.POINTER(C.c_int32))]


class Packing(C.Structure):         # imd_packing_t
    _fields_ = [("A", C.POINTER(PackDesc)), ("B", C.POINTER(PackDesc))]


class Operand(C.Structure):         # imd_operand_t
    _fields_ = [("data", C.c_void_p), ("packed", C.c_void_p), ("packed_bytes", C.c_size_t),
                ("pack", C.c_void_p)]


assert C.sizeof(Tensor) == 32 and C.sizeof(Options) == 24
assert C.sizeof(PackDesc) == 72 and C.sizeof(Packing) == 16 and C.sizeof(Operand) == 32

_T, _O, _P = C.POINTER(Tensor), C.POINTER(Options), C.c_void_p
_SZ = C.POINTER(C.c_size_t)
# name -> (restype, argtypes): every function of immerdink.h, plus the host functions
# (immerdink_host.h, reached through libimmerdink's DT_NEEDED) the device API uses.
SIGNATURES = {
    "imd_create": (C.c_int, [C.POINTER(C.c_void_p)]),
    "imd_destroy": (C.c_int, [_P]),
    "imd_plan_create": (C.c_int, [_P, _T, _T, _T, _O, C.POINTER(C.c_void_p)]),
    "imd_plan_destroy": (C.c_int, [_P]),
    "imd_plan_device_bytes": (C.c_int, [_P, _SZ]),
    "imd_contract": (C.c_int, [_P, C.c_double, _P, _P, C.c_double, _P, _P, C.c_size_t, _P]),
    "imd_pack_create": (C.c_int, [_P, C.POINTER(PackDesc), C.POINTER(C.c_void_p)]),
    "imd_pack_destroy": (C.c_int, [_P]),
    "imd_pack_bytes": (C.c_int, [_P, _SZ, _SZ]),
    "imd_pack": (C.c_int, [_P, _P, _P, C.c_size_t, _P, C.c_size_t, _P]),
    "imd_plan_create_packed": (C.c_int, [_P, _T, _T, _T, _O, _P, _P, C.POINTER(C.c_void_p)]),
    "imd_contract_packed": (C.c_int, [_P, C.c_double, C.POINTER(Operand), C.POINTER(Operand),
                                      C.c_double, _P, _P, C.c_size_t, _P]),
    # immerdink_host.h
    "imd_host_api_version": (C.c_uint32, []),
    "imd_status_string": (C.c_char_p, [C.c_int]),
    "imd_workspace_size": (C.c_int, [_T, _T, _T, _O, _SZ]),
    "imd_workspace_size_packed": (C.c_int, [_T, _T, _T, _O, C.POINTER(Packing), _SZ]),
    "imd_plan_bytes": (C.c_int, [_T, _T, _T, _O, _SZ, _SZ]),
}


def _candidates():
    if os.environ.get("IMMERDINK_LIB"):
        yield os.environ["IMMERDINK_LIB"]
    if os.environ.get("IMMERDINK_LIB_DIR"):
        yield os.path.join(os.environ["IMMERDINK_LIB_DIR"], "libimmerdink.so")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build = os.environ.get("IMMERDINK_BUILD_DIR") or os.path.join(repo, "build")
    yield os.path.join(build, "libimmerdink.so")
    found = ctypes.util.find_library("immerdink")
    if found:
        yield found


class Library:
    """libimmerdink (ctypes), its host library (host.HostLibrary) and cudaGetDevice from the
    libcudart it loaded."""

    def __init__(self, path=None):
        if path is None:
            tried = []
            for cand in _candidates():
                tried.append(cand)
                if os.path.sep not in cand or os.path.exists(cand):
                    path = cand
                    break
            if path is None:
                raise OSError("libimmerdink not found (set IMMERDINK_LIB or IMMERDINK_LIB_DIR); "
                              f"tried {tried}")
        self.path = path
        lib = self.lib = C.CDLL(path)
        for name, (res, args) in SIGNATURES.items():
            f = getattr(lib, name)
            f.restype, f.argtypes = res, args
        self.version = lib.imd_host_api_version()
        if self.version < API_VERSION:
            raise OSError(f"{path}: host API version {self.version} < {API_VERSION}")
        # The host library libimmerdink links (the sibling file when there is one), so host
        # queries report the numbers this library uses.
        sib = os.path.join(os.path.dirname(os.path.abspath(path)), "libimmerdink_host.so")
        self.host = host.load(sib if os.path.sep in path and os.path.exists(sib) else None)
        self.cudart = None
        for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
            try:  # already loaded by libimmerdink: dlopen returns that copy
                self.cudart = C.CDLL(name)
                break
            except OSError:
                continue
        if self.cudart is not None:
            self.cudart.cudaGetDevice.argtypes = [C.POINTER(C.c_int)]
            self.cudart.cudaGetDevice.restype = C.c_int
            self.cudart.cudaDeviceGetAttribute.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
            self.cudart.cudaDeviceGetAttribute.restype = C.c_int

    def current_device(self):
        """cudaGetDevice (host-only, capture-safe); 0 when libcudart cannot be found."""
        if self.cudart is None:
            return 0
        d = C.c_int(0)
        err = self.cudart.cudaGetDevice(C.byref(d))
        if err != 0:
            raise RuntimeError(f"cudaGetDevice failed ({err})")
        return d.value

    def sm_count(self, device=None):
        """The SM count (cudaDevAttrMultiProcessorCount) of `device` (None: the current one);
        0 (the host library's default) when libcudart cannot be found."""
        if self.cudart is None:
            return 0
        v = C.c_int(0)
        dev = self.current_device() if device is None else device
        if self.cudart.cudaDeviceGetAttribute(C.byref(v), 16, dev) != 0:
            raise RuntimeError("cudaDeviceGetAttribute failed")
        return v.value


_loaded = {}
_lock = threading.Lock()


def load(path=None):
    """The libimmerdink at `path` (None: the default search), loaded once."""
    lib = _loaded.get(path)
    if lib is None:
        with _lock:
            lib = _loaded.get(path)
            if lib is None:
                lib = _loaded[path] = Library(path)
    return lib
