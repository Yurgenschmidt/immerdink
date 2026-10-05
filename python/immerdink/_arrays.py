"""CUDA array protocols for the device API (DESIGN §6c.3): __cuda_array_interface__ (v2/v3),
DLPack (0.x and 1.x capsules) and raw device pointers (DeviceArray), plus stream objects.
Pure Python over ctypes: no CUDA call, no allocation, no array library."""

import ctypes as C
import math
import struct

__all__ = ["DeviceArray", "view", "buffer", "stream_handle"]

KDL_CUDA, KDL_CUDA_MANAGED = 2, 13
_DL_FLOAT = 2
_DL_READ_ONLY = 1  # DLPack 1.x DLPACK_FLAG_BITMASK_READ_ONLY


class DeviceArray:
    """A raw device pointer as an array: `ptr` is the address of element (0, ..., 0),
    `strides` are in ELEMENTS (None: dense, last mode fastest; any sign), `dtype` a numpy-style
    type string ("f8" = float64, "u1" = bytes, ...). Exports __cuda_array_interface__ v3;
    immerdink never dereferences it on the host and never frees it."""

    __slots__ = ("ptr", "shape", "strides", "dtype", "readonly", "__cuda_array_interface__")

    def __init__(self, ptr, shape, strides=None, dtype="f8", readonly=False):
        self.ptr = int(ptr.value if isinstance(ptr, C.c_void_p) else ptr or 0)
        self.shape = tuple(int(e) for e in shape)
        dtype = dtype.lstrip("<|=")
        itemsize = int(dtype[1:])
        self.dtype, self.readonly = dtype, bool(readonly)
        if strides is None:
            self.strides = None
            bstrides = None
        else:
            self.strides = tuple(int(s) for s in strides)
            if len(self.strides) != len(self.shape):
                raise ValueError(f"{len(self.strides)} strides for {len(self.shape)} modes")
            bstrides = tuple(s * itemsize for s in self.strides)
        self.__cuda_array_interface__ = {
            "shape": self.shape, "typestr": ("|" if itemsize == 1 else "<") + dtype,
            "data": (self.ptr, self.readonly), "strides": bstrides, "version": 3,
            "stream": None}

    @property
    def nbytes(self):
        return math.prod(self.shape) * int(self.dtype[1:])

    def __repr__(self):
        return (f"DeviceArray(0x{self.ptr:x}, shape={self.shape}, strides={self.strides}, "
                f"dtype={self.dtype!r})")


# ------------------------------------------------------------------------------------------
# DLPack (dlpack.h)

class _DLDevice(C.Structure):
    _fields_ = [("device_type", C.c_int32), ("device_id", C.c_int32)]


class _DLDataType(C.Structure):
    _fields_ = [("code", C.c_uint8), ("bits", C.c_uint8), ("lanes", C.c_uint16)]


class _DLTensor(C.Structure):
    _fields_ = [("data", C.c_void_p), ("device", _DLDevice), ("ndim", C.c_int32),
                ("dtype", _DLDataType), ("shape", C.POINTER(C.c_int64)),
                ("strides", C.POINTER(C.c_int64)), ("byte_offset", C.c_uint64)]


class _DLManagedTensor(C.Structure):          # DLPack 0.x capsule "dltensor"
    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", C.c_void_p), ("deleter", C.c_void_p)]


class _DLPackVersion(C.Structure):
    _fields_ = [("major", C.c_uint32), ("minor", C.c_uint32)]


class _DLManagedTensorVersioned(C.Structure):  # DLPack 1.x capsule "dltensor_versioned"
    _fields_ = [("version", _DLPackVersion), ("manager_ctx", C.c_void_p),
                ("deleter", C.c_void_p), ("flags", C.c_uint64), ("dl_tensor", _DLTensor)]


_api = C.pythonapi
_api.PyCapsule_IsValid.argtypes = [C.py_object, C.c_char_p]
_api.PyCapsule_IsValid.restype = C.c_int
_api.PyCapsule_GetPointer.argtypes = [C.py_object, C.c_char_p]
_api.PyCapsule_GetPointer.restype = C.c_void_p
_api.PyCapsule_GetName.argtypes = [C.py_object]
_api.PyCapsule_GetName.restype = C.c_char_p
_u64 = C.c_uint64
_DLT = struct.Struct("<QiiiBBHQQQ")             # DLTensor, 48 bytes (x86-64 / aarch64)
_I64 = [struct.Struct(f"<{n}q") for n in range(65)]
assert _DLT.size == C.sizeof(_DLTensor) == 48


def _dlpack_view(x, stream):
    """(ptr, shape, byte strides or None, is_f8, readonly, itemsize) of a DLPack producer."""
    dev = x.__dlpack_device__()
    if int(dev[0]) not in (KDL_CUDA, KDL_CUDA_MANAGED):
        raise TypeError(f"{type(x).__name__} lives on DLPack device type {int(dev[0])}, not CUDA "
                        "memory (immerdink never copies host <-> device)")
    s = 1 if stream in (None, 0) else stream   # DLPack: 1 = the legacy default stream
    try:
        cap = x.__dlpack__(stream=s, max_version=(1, 0))
    except TypeError:                          # a DLPack 0.x producer
        cap = x.__dlpack__(stream=s)
    name = _api.PyCapsule_GetName(cap)
    if name == b"dltensor_versioned":
        # DLManagedTensorVersioned: version (8), manager_ctx, deleter, flags, dl_tensor
        p = _api.PyCapsule_GetPointer(cap, name)
        readonly = bool(_u64.from_address(p + 24).value & _DL_READ_ONLY)
        p += 32
    elif name == b"dltensor":
        p, readonly = _api.PyCapsule_GetPointer(cap, name), False
    else:
        raise TypeError(f"{type(x).__name__}.__dlpack__ returned no DLPack capsule")
    # DLTensor: data, device (2 x int32), ndim, dtype (u8 code, u8 bits, u16 lanes), shape,
    # strides, byte_offset
    data, _, _, nd, code, bits, lanes, pshape, pstrides, off = _DLT.unpack(C.string_at(p, 48))
    i64 = _I64[nd] if nd < len(_I64) else struct.Struct(f"<{nd}q")
    shape = i64.unpack(C.string_at(pshape, 8 * nd)) if nd else ()
    itemsize = bits * lanes // 8
    strides = None if not pstrides else \
        tuple(v * itemsize for v in i64.unpack(C.string_at(pstrides, 8 * nd)))
    ptr = data + off
    f8 = code == _DL_FLOAT and bits == 64 and lanes == 1
    # The capsule is dropped here: its destructor calls the producer's deleter (immerdink
    # takes no ownership; the caller keeps x alive while its GPU work runs).
    return ptr, shape, strides, f8, readonly, itemsize


def view(x, stream=None):
    """(ptr, shape, byte strides or None (C-contiguous), is_float64, readonly, itemsize) of a
    CUDA array: __cuda_array_interface__ first, else DLPack."""
    try:
        d = x.__cuda_array_interface__
    except AttributeError:
        if hasattr(x, "__dlpack__"):
            return _dlpack_view(x, stream)
        raise TypeError(f"{type(x).__name__} is not a CUDA array (no __cuda_array_interface__ "
                        "or __dlpack__); immerdink never copies host <-> device") from None
    if d.get("mask") is not None:
        raise TypeError("masked arrays are not supported")
    ts = d["typestr"]
    data = d["data"]
    return data[0] or 0, d["shape"], d.get("strides"), ts == "<f8", data[1], int(ts[2:])


def buffer(x, what, writable=True, stream=None):
    """(ptr, nbytes) of a byte buffer: None, an (int ptr, nbytes) tuple, or a contiguous CUDA
    array of any dtype."""
    if x is None:
        return None, 0
    if isinstance(x, tuple):
        return int(x[0] or 0), int(x[1])
    ptr, shape, strides, _, readonly, itemsize = view(x, stream)
    if writable and readonly:
        raise ValueError(f"{what} is read-only")
    n = math.prod(shape)
    if strides is not None and n > 1:
        acc = itemsize
        for e, s in zip(reversed(shape), reversed(strides)):
            if e > 1 and s != acc:
                raise ValueError(f"{what} must be contiguous")
            acc *= e
    return ptr, n * itemsize


def stream_handle(s):
    """The cudaStream_t value of a stream argument (DESIGN §6c.5): None / 0 = NULL (the
    legacy default stream), an int as is (1 = cudaStreamLegacy, 2 = cudaStreamPerThread),
    __cuda_stream__ (CUDA Python protocol), .cuda_stream (PyTorch), .ptr (CuPy), c_void_p."""
    if s is None:
        return 0
    if isinstance(s, int):
        return s
    if hasattr(s, "__cuda_stream__"):
        return int(s.__cuda_stream__()[1])
    for attr in ("cuda_stream", "ptr"):
        h = getattr(s, attr, None)
        if isinstance(h, int):
            return h
    if isinstance(s, C.c_void_p):
        return s.value or 0
    raise TypeError(f"not a CUDA stream: {type(s).__name__} (expected an int handle or an "
                    "object with __cuda_stream__, .cuda_stream or .ptr)")
