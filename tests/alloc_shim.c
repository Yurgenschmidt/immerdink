// LD_PRELOAD allocation counter for the "imd_contract allocates nothing" test (DESIGN §6,
// PLAN M3). Interposes the CUDA runtime's allocation entry points, counts every call and
// forwards it to the real libcudart (dlsym RTLD_NEXT). tests/test_frontend.py reads the
// count before and after imd_contract. Test-only; never linked into the library.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime_api.h>

#ifndef IMD_CUDART_SONAME  // set by tests/CMakeLists.txt from the toolkit version
#define IMD_CUDART_SONAME "libcudart.so.13"
#endif

static int64_t g_count = 0;

int64_t imd_shim_alloc_count(void) { return __atomic_load_n(&g_count, __ATOMIC_SEQ_CST); }

// tests/test_contract.py proves interposition by observing the one cudaMalloc of an
// IMD_VEC_RESIDENT imd_plan_create (DESIGN v1.2) in the count.

// The real entry point: the next definition in the global scope, or (when libcudart was
// loaded RTLD_LOCAL, as Python's ctypes does) libcudart's own definition.
static void* real_symbol(const char* name) {
  void* f = dlsym(RTLD_NEXT, name);
  if (f != NULL) return f;
  void* h = dlopen(IMD_CUDART_SONAME, RTLD_NOW | RTLD_NOLOAD);
  if (h == NULL) h = dlopen(IMD_CUDART_SONAME, RTLD_NOW);
  return h == NULL ? NULL : dlsym(h, name);
}

#define FORWARD(name, params, args)                    \
  cudaError_t name params {                            \
    typedef cudaError_t(*fn_t) params;                 \
    static fn_t real = NULL;                           \
    if (real == NULL) real = (fn_t)real_symbol(#name); \
    __atomic_add_fetch(&g_count, 1, __ATOMIC_SEQ_CST); \
    return real == NULL ? cudaErrorUnknown : real args; \
  }

FORWARD(cudaMalloc, (void** p, size_t n), (p, n))
FORWARD(cudaMallocHost, (void** p, size_t n), (p, n))
FORWARD(cudaHostAlloc, (void** p, size_t n, unsigned int f), (p, n, f))
FORWARD(cudaMallocManaged, (void** p, size_t n, unsigned int f), (p, n, f))
FORWARD(cudaMallocPitch, (void** p, size_t* pitch, size_t w, size_t h), (p, pitch, w, h))
FORWARD(cudaMallocAsync, (void** p, size_t n, cudaStream_t s), (p, n, s))
FORWARD(cudaMallocFromPoolAsync, (void** p, size_t n, cudaMemPool_t m, cudaStream_t s),
        (p, n, m, s))
