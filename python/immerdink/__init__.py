"""immerdink from Python.

- `immerdink.host`: exact memory footprints and the lowering summary from the CUDA-free
  libimmerdink_host (DESIGN §6a), no GPU needed.
- `immerdink.planner`: the same host queries for hot paths (DESIGN §6e): `Query`, `Planner`,
  opt-in `Memo`, lazy results; bit-identical to `host`, about 1-2 µs per query.
- The device API (DESIGN §6d, immerdink.device): `Plan`, `Pack`, `contract` (einsum front
  end) on caller-owned float64 CUDA arrays (__cuda_array_interface__, DLPack or
  `DeviceArray` raw pointers), a ctypes binding of libimmerdink's C ABI. Importing loads no
  library; libimmerdink is loaded on first use.
- Precision phases (DESIGN v1.19 §6g): a plan and a pack per N; `PackSpec.with_moduli` and
  `PackedBuffer` (a packed buffer re-packed at several N, guarded on the host);
  `immerdink.precision` (not imported here): `PrecisionPlan`, N per phase from targets on the
  host, and `rho_from_output`, the rigorous rho of a previous output (needs numpy).
"""

from .device import (  # noqa: F401
    AliasingError, AllocFailedError, CudaError, DeviceArray, Handle, ImmerdinkError,
    InsufficientWorkspaceError, InvalidValueError, Layout, MisalignedError, NotSupportedError,
    Pack, PackedBuffer, PackMismatchError, PackRecord, Plan, PlanCache, contract,
    default_handle, default_plan_cache, get_cai_stream_policy, load_library, pack_spec,
    set_cai_stream_policy, workspace_size)
from .host import PackSpec  # noqa: F401
