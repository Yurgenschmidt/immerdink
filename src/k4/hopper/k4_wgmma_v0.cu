// Hopper K4 variant 0 (host/k4_wgmma.hpp): CTA tile 128 x 256 x 128, cluster 2 x 1, cooperative. See k4_wgmma_impl.cuh.
#define IMD_K4W_NAME v0
#define IMD_K4W_TILE cute::Shape<cute::Int<128>, cute::Int<256>, cute::Int<128>>
#define IMD_K4W_CLUSTER cute::Shape<cute::Int<2>, cute::Int<1>, cute::Int<1>>
#define IMD_K4W_SCHED cutlass::gemm::KernelTmaWarpSpecializedCooperative
#include "k4/hopper/k4_wgmma_impl.cuh"
