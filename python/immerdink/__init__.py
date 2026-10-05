"""immerdink from Python.

- `immerdink.host`: exact memory footprints and the lowering summary from the CUDA-free
  libimmerdink_host (DESIGN §6a), no GPU needed.
- The device API (DESIGN §6d, immerdink.device): `Plan`, `Pack`, `contract` (einsum front
  end) on caller-owned float64 CUDA arrays (__cuda_array_interface__, DLPack or
  `DeviceArray` raw pointers), a ctypes binding of libimmerdink's C ABI. Importing loads no
  library; libimmerdink is loaded on first use.
"""

from .device import (  # noqa: F401
    AliasingError, AllocFailedError, CudaError, DeviceArray, Handle, ImmerdinkError,
    InsufficientWorkspaceError, InvalidValueError, Layout, MisalignedError, NotSupportedError,
    Pack, PackMismatchError, Plan, PlanCache, contract, default_handle, default_plan_cache,
    get_cai_stream_policy, load_library, pack_spec, set_cai_stream_policy, workspace_size)
from .host import PackSpec  # noqa: F401
