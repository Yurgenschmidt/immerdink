// Per-device classification (src/device_state.cpp, issue 18). Thread-safe; the result is cached.
#pragma once

#include <cuda_runtime_api.h>

#include "immerdink/immerdink_host.h"

namespace imd::detail {

// The device class, the table-set selection and the code kind of every kernel of the library on
// `device` (cudaFuncGetAttributes on all of them: this also loads them, so a missing image or PTX
// that needs a newer driver shows here and not at the first imd_contract). out->k4_config is -1.
// Returns the CUDA error if the device could not be examined (not cached), else cudaSuccess: an
// unusable device is a result (status UNSUPPORTED, failing_kernel, cuda_error), not an error.
cudaError_t device_state(int device, imd_device_info_t* out);

}  // namespace imd::detail
