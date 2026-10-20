#pragma once

#include <cstdint>
#include <stdio.h>

#define CUDA_CHECK(cmd)                                             \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

int64_t get_device_attribute(int64_t attribute, int64_t device_id);

int64_t get_max_shared_memory_per_block_device_attribute(int64_t device_id);

// gfx906 fork: restored after the v0.32.0 merge. Upstream rewrote this header
// and dropped the fork's `cuda_utils::ceil_div`, which csrc/cache_kernels.cu
// (our merged split-KV insert path) still calls. Dropped silently -- no conflict
// marker -- because upstream's rewrite landed in a file we had also modified.
// Upstream also dropped the HOST_DEVICE_INLINE macro in 0.32.0, so the
// qualifier is spelled out here.
namespace cuda_utils {

// This header is included from host-only TUs as well (no CUDA macros there),
// so the device qualifier is conditional rather than unconditional.
#if defined(__CUDACC__) || defined(__HIPCC__)
  #define VLLM_CEIL_DIV_QUALIFIER __host__ __device__
#else
  #define VLLM_CEIL_DIV_QUALIFIER
#endif

template <typename T>
VLLM_CEIL_DIV_QUALIFIER constexpr std::enable_if_t<std::is_integral_v<T>, T>
ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

};  // namespace cuda_utils

#undef VLLM_CEIL_DIV_QUALIFIER
