// G2b native comparators (immerdink-eval analysis/20261110-dataflow/plan.md §4.2 and §6 step 3), extended from
// tmp/oz/dsyrk_bench.cu (same timing rule: interleaved call by call, medians). Self-contractions C = A A^T with A: n x k,
// FP64, values uniform [-1, 1), column-major (cuBLAS), and an R1-like permuted output aiQ,bjQ->abij.
// TFLOPS-eq = the DENSE count 2 n^2 k / median time for every method (so a half-work SYRK can show up to 2x).
//
// Cases: t_df (n = 4096, k = 2048), r_lowk (n = 8192, k = 1024): matrix output C[i + j n];
//        r1_q1024 / r1_q2048: X_aiQ with a = b = 128, i = j = 32 (rows r = 32 a + i, n = 4096), Q = 1024 / 2048,
//        output C_abij row-major (j fastest): C[((a 128 + b) 32 + i) 32 + j] = sum_Q X_aiQ X_bjQ.
// Methods (complete = the full result in its final layout):
//   matrix: dsyrk (lower only, NOT complete), mirror (alone, on a filled lower), dsyrk+mirror, dsyrkx+mirror, dgemm,
//           emu55 (cublasGemmEx, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, FIXED 55 bits, EAGER: as immerdink-eval bench/eval_bench.cu
//           method_emu), emumath-dgemm and emumath-dsyrk+mirror (cublasDgemm / cublasDsyrk on a handle with math mode
//           CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH, FIXED 55, EAGER: does the plain API route through emulation?), cutensor
//           (ik,jk->ij, CUTENSOR_COMPUTE_DESC_64F, as eval_bench method_cutensor)
//   r1:     dsyrk (into scratch G, NOT complete), mirror_permute (alone), dsyrk+mirror_permute, dgemm+permute, emu55+permute,
//           emumath-dsyrk+mirror_permute, cutensor (aiQ,bjQ->abij directly), permute (alone)
// Checks (outside the timed loop): every complete method against dgemm (+ permute): max |diff| / max |ref|; and the emulation
// routing evidence: bit differences of emumath-dsyrk vs dsyrk and vs emu55 (lower triangle), emumath-dgemm vs dgemm and vs emu55.
// --mode probe: n = 2048, k = 1024, one call each of dsyrk, emumath-dsyrk, emumath-dgemm, emu55, dsyrkx, emumath-dsyrkx with a
// marker kernel (probe_marker) between them, for `ncu --metrics gpu__time_duration.sum` to list the kernel names each runs.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cutensor.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { std::printf("cuBLAS %d line %d\n", int(s_), __LINE__); std::exit(1); } } while (0)
#define CT(x) do { cutensorStatus_t s_ = (x); if (s_ != CUTENSOR_STATUS_SUCCESS) { std::printf("cuTENSOR %s line %d\n", cutensorGetErrorString(s_), __LINE__); std::exit(1); } } while (0)

// ---- kernels ------------------------------------------------------------------------------------------------------------
// In-place mirror of a column-major n x n matrix whose lower triangle (row >= col) is filled: upper(r, c) = lower(c, r).
// 32 x 32 tiles, block (32, 8); block (bi, bj) with bi >= bj reads lower tile (bi, bj) coalesced along rows and writes the
// transposed tile (bj, bi) coalesced along rows; diagonal tiles write only row < col. n % 32 == 0.
__global__ void mirror_lower(double* C, int n) {
  const int bi = blockIdx.y, bj = blockIdx.x;
  if (bj > bi) return;
  __shared__ double s[32][33];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const size_t R0 = size_t(bi) * 32, S0 = size_t(bj) * 32;
  for (int c = ty; c < 32; c += 8) s[c][tx] = C[(R0 + tx) + (S0 + c) * n];   // s[c][r] = C(R0 + r, S0 + c)
  __syncthreads();
  for (int r = ty; r < 32; r += 8) {   // upper element (row S0 + tx, col R0 + r) = C(R0 + r, S0 + tx) = s[tx][r]
    if (bi != bj || tx < r) C[(S0 + tx) + (R0 + r) * n] = s[tx][r];
  }
}
// R1 output from the LOWER triangle of G (column-major n x n, n = 32 A): C[((a A + b) 32 + i) 32 + j] = G(32 a + i, 32 b + j),
// read as G(32 max(a,b) + .., 32 min(a,b) + ..) of the lower triangle. Block (b, a), threads (32, 8): reads coalesced along the
// lower tile's rows, writes coalesced along j.
__global__ void mirror_permute(const double* G, double* C, int n, int A) {
  const int a = blockIdx.y, b = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  __shared__ double s[32][33];
  const int hi = a > b ? a : b, lo = a > b ? b : a;
  const size_t R0 = size_t(hi) * 32, S0 = size_t(lo) * 32;
  for (int c = ty; c < 32; c += 8) s[c][tx] = G[(R0 + tx) + (S0 + c) * n];   // s[c][r] = G(R0 + r, S0 + c)
  __syncthreads();
  double* out = C + (size_t(a) * A + b) * 1024;
  for (int i = ty; i < 32; i += 8) {
    const int j = tx;
    double v;
    if (a > b) v = s[j][i];                        // G(32a + i, 32b + j): r = i, c = j
    else if (a < b) v = s[i][j];                   // G(32b + j, 32a + i): r = j, c = i
    else v = i >= j ? s[j][i] : s[i][j];
    out[i * 32 + j] = v;
  }
}
// R1 output from a FULL G (dense GEMM result): C[((a A + b) 32 + i) 32 + j] = G(32 a + i, 32 b + j).
__global__ void permute_full(const double* G, double* C, int n, int A) {
  const int a = blockIdx.y, b = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;
  __shared__ double s[32][33];
  const size_t R0 = size_t(a) * 32, S0 = size_t(b) * 32;
  for (int c = ty; c < 32; c += 8) s[c][tx] = G[(R0 + tx) + (S0 + c) * n];   // s[c][r] = G(32a + r, 32b + c)
  __syncthreads();
  double* out = C + (size_t(a) * A + b) * 1024;
  for (int i = ty; i < 32; i += 8) out[i * 32 + tx] = s[tx][i];
}
// x vs y over a column-major n x n region (lower = row >= col only, else all): bit differences, max |x - y|, max |y|.
__global__ void cmp_region(const double* x, const double* y, int n, int lower, unsigned long long* res) {
  unsigned long long cnt = 0, md = 0, my = 0;
  const size_t total = size_t(n) * n;
  for (size_t q = blockIdx.x * size_t(blockDim.x) + threadIdx.x; q < total; q += size_t(gridDim.x) * blockDim.x) {
    const size_t r = q % n, c = q / n;
    if (lower && r < c) continue;
    const double a = x[q], b = y[q];
    cnt += __double_as_longlong(a) != __double_as_longlong(b);
    md = max(md, (unsigned long long)__double_as_longlong(fabs(a - b)));
    my = max(my, (unsigned long long)__double_as_longlong(fabs(b)));
  }
  atomicAdd(&res[0], cnt); atomicMax(&res[1], md); atomicMax(&res[2], my);
}
__global__ void probe_marker(int id) { if (id < 0) printf("%d\n", id); }

// ---- host ---------------------------------------------------------------------------------------------------------------
struct Case { std::string name; int n, k, r1A; };   // r1A > 0: R1 case with a = b = r1A, i = j = 32 (n = 32 r1A)
struct Method { std::string name; bool complete; std::function<void()> run; double* out; int lower_only; };
struct Cmp { unsigned long long bits; double maxdiff, maxref; };

static Cmp compare(const double* x, const double* y, int n, bool lower, unsigned long long* dres) {
  CK(cudaMemset(dres, 0, 24));
  cmp_region<<<1024, 256>>>(x, y, n, lower ? 1 : 0, dres);
  CK(cudaDeviceSynchronize());
  unsigned long long h[3]; CK(cudaMemcpy(h, dres, 24, cudaMemcpyDeviceToHost));
  Cmp c; c.bits = h[0];
  double d, m; std::memcpy(&d, &h[1], 8); std::memcpy(&m, &h[2], 8);
  c.maxdiff = d; c.maxref = m;
  return c;
}

int main(int argc, char** argv) {
  std::string mode = "all", out_path, cases_arg = "t_df,r_lowk,r1_q1024,r1_q2048";
  int reps = 41, emu_ws_mib = 4096;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--mode") mode = v; else if (k == "--out") out_path = v; else if (k == "--reps") reps = std::atoi(v.c_str());
    else if (k == "--cases") cases_arg = v; else if (k == "--emu-ws-mib") emu_ws_mib = std::atoi(v.c_str());
    else { std::printf("unknown option %s\n", k.c_str()); return 2; }
  }
  FILE* out = out_path.empty() ? nullptr : std::fopen(out_path.c_str(), "a");
  auto emit = [&](const char* line) { std::fputs(line, stdout); if (out) { std::fputs(line, out); std::fflush(out); } };
  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  int cublas_ver = 0;
  cudaStream_t st = 0;
  cublasHandle_t h, eh, sh;   // native; GemmEx emulation (eval_bench method_emu); math-mode emulation (plain API routing)
  CB(cublasCreate(&h)); CB(cublasCreate(&eh)); CB(cublasCreate(&sh));
  CB(cublasGetVersion(h, &cublas_ver));
  void *ews, *sws;
  const size_t wsb = size_t(emu_ws_mib) << 20;
  CK(cudaMalloc(&ews, wsb)); CK(cudaMalloc(&sws, wsb));
  for (cublasHandle_t x : {eh, sh}) {
    CB(cublasSetWorkspace(x, x == eh ? ews : sws, wsb));
    CB(cublasSetEmulationStrategy(x, CUBLAS_EMULATION_STRATEGY_EAGER));
    CB(cublasSetFixedPointEmulationMantissaControl(x, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
    CB(cublasSetFixedPointEmulationMaxMantissaBitCount(x, 55));
  }
  CB(cublasSetMathMode(sh, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
  {
    int mb = -1; cudaEmulationMantissaControl mc{}; cublasMath_t mm{};
    cublasGetFixedPointEmulationMaxMantissaBitCount(sh, &mb); cublasGetFixedPointEmulationMantissaControl(sh, &mc); cublasGetMathMode(sh, &mm);
    char line[512];
    std::snprintf(line, sizeof line, "{\"meta\":1,\"device\":\"%s\",\"sms\":%d,\"cublas\":%d,\"cutensor\":%zu,\"emu_max_bits\":%d,\"emu_control_fixed\":%d,"
                  "\"emumath_mode\":%d,\"emu_ws_mib\":%d,\"reps\":%d,\"mode\":\"%s\"}\n", prop.name, prop.multiProcessorCount, cublas_ver,
                  size_t(cutensorGetVersion()), mb, int(mc == CUDA_EMULATION_MANTISSA_CONTROL_FIXED), int(mm), emu_ws_mib, reps, mode.c_str());
    emit(line);
  }
  cutensorHandle_t th; CT(cutensorCreate(&th));
  const double one = 1.0, zero = 0.0;
  unsigned long long* dres; CK(cudaMalloc(&dres, 24));
  std::mt19937_64 rng(1); std::uniform_real_distribution<double> u(-1.0, 1.0);

  if (mode == "probe") {   // kernel names per call, for ncu
    const int n = 2048, k = 1024;
    std::vector<double> hA(size_t(n) * k); for (auto& x : hA) x = u(rng);
    double *A, *C; CK(cudaMalloc(&A, hA.size() * 8)); CK(cudaMalloc(&C, size_t(n) * n * 8));
    CK(cudaMemcpy(A, hA.data(), hA.size() * 8, cudaMemcpyHostToDevice));
    probe_marker<<<1, 1>>>(1); CB(cublasDsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one, A, n, &zero, C, n));
    probe_marker<<<1, 1>>>(2); CB(cublasDsyrk(sh, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one, A, n, &zero, C, n));
    probe_marker<<<1, 1>>>(3); CB(cublasDgemm(sh, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &one, A, n, A, n, &zero, C, n));
    probe_marker<<<1, 1>>>(4); CB(cublasGemmEx(eh, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &one, A, CUDA_R_64F, n, A, CUDA_R_64F, n, &zero, C, CUDA_R_64F, n,
                                               CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT));
    probe_marker<<<1, 1>>>(5); CB(cublasDsyrkx(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one, A, n, A, n, &zero, C, n));
    probe_marker<<<1, 1>>>(6); CB(cublasDsyrkx(sh, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one, A, n, A, n, &zero, C, n));
    probe_marker<<<1, 1>>>(7); CK(cudaDeviceSynchronize());
    std::printf("# probe done: markers 1 dsyrk | 2 emumath-dsyrk | 3 emumath-dgemm | 4 emu55 GemmEx | 5 dsyrkx | 6 emumath-dsyrkx | 7 end\n");
    return 0;
  }

  std::vector<Case> cases;
  for (const Case& c : {Case{"t_df", 4096, 2048, 0}, Case{"r_lowk", 8192, 1024, 0}, Case{"r1_q1024", 4096, 1024, 128}, Case{"r1_q2048", 4096, 2048, 128}})
    if (("," + cases_arg + ",").find("," + c.name + ",") != std::string::npos) cases.push_back(c);
  for (const Case& cs : cases) {
    const int n = cs.n, k = cs.k;
    const bool r1 = cs.r1A > 0;
    const size_t na = size_t(n) * k, nc = size_t(n) * n;
    std::vector<double> hA(na); for (auto& x : hA) x = u(rng);
    double *A, *C, *Cref, *G = nullptr, *T1, *T2;   // T1 / T2: outputs of the routing probes
    CK(cudaMalloc(&A, na * 8)); CK(cudaMalloc(&C, nc * 8)); CK(cudaMalloc(&Cref, nc * 8)); CK(cudaMalloc(&T1, nc * 8)); CK(cudaMalloc(&T2, nc * 8));
    if (r1) CK(cudaMalloc(&G, nc * 8));
    CK(cudaMemcpy(A, hA.data(), na * 8, cudaMemcpyHostToDevice)); CK(cudaMemset(C, 0, nc * 8));
    // cuTENSOR contraction
    cutensorTensorDescriptor_t da, db, dc;
    std::vector<int32_t> ma, mb, mc;
    if (!r1) {
      std::vector<int64_t> ea{n, k}, sa{1, n}, ec{n, n}, sc{1, n};
      CT(cutensorCreateTensorDescriptor(th, &da, 2, ea.data(), sa.data(), CUTENSOR_R_64F, 256));
      CT(cutensorCreateTensorDescriptor(th, &db, 2, ea.data(), sa.data(), CUTENSOR_R_64F, 256));
      CT(cutensorCreateTensorDescriptor(th, &dc, 2, ec.data(), sc.data(), CUTENSOR_R_64F, 256));
      ma = {'i', 'k'}; mb = {'j', 'k'}; mc = {'i', 'j'};
    } else {
      const int64_t Ae = cs.r1A;
      std::vector<int64_t> ea{32, Ae, k}, sa{1, 32, n}, ec{32, 32, Ae, Ae}, sc{1, 32, 1024, 1024 * Ae};
      CT(cutensorCreateTensorDescriptor(th, &da, 3, ea.data(), sa.data(), CUTENSOR_R_64F, 256));   // X(i, a, Q): row r = 32 a + i
      CT(cutensorCreateTensorDescriptor(th, &db, 3, ea.data(), sa.data(), CUTENSOR_R_64F, 256));   // X(j, b, Q)
      CT(cutensorCreateTensorDescriptor(th, &dc, 4, ec.data(), sc.data(), CUTENSOR_R_64F, 256));   // C(j, i, b, a): abij row-major
      ma = {'i', 'a', 'Q'}; mb = {'j', 'b', 'Q'}; mc = {'j', 'i', 'b', 'a'};
    }
    cutensorOperationDescriptor_t op;
    CT(cutensorCreateContraction(th, &op, da, ma.data(), CUTENSOR_OP_IDENTITY, db, mb.data(), CUTENSOR_OP_IDENTITY, dc, mc.data(),
                                 CUTENSOR_OP_IDENTITY, dc, mc.data(), CUTENSOR_COMPUTE_DESC_64F));
    cutensorPlanPreference_t pref; CT(cutensorCreatePlanPreference(th, &pref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
    uint64_t ctw = 0; CT(cutensorEstimateWorkspaceSize(th, op, pref, CUTENSOR_WORKSPACE_DEFAULT, &ctw));
    cutensorPlan_t plan; CT(cutensorCreatePlan(th, &plan, op, pref, ctw));
    void* ctws = nullptr; if (ctw) CK(cudaMalloc(&ctws, ctw));

    const dim3 tb(32, 8), tg(n / 32, n / 32), pg(cs.r1A, cs.r1A);
    auto syrk = [&](cublasHandle_t hh, double* dst) { CB(cublasDsyrk(hh, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one, A, n, &zero, dst, n)); };
    auto syrkx = [&](double* dst) { CB(cublasDsyrkx(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &one, A, n, A, n, &zero, dst, n)); };
    auto gemm = [&](cublasHandle_t hh, double* dst) { CB(cublasDgemm(hh, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &one, A, n, A, n, &zero, dst, n)); };
    auto emu = [&](double* dst) {
      CB(cublasGemmEx(eh, CUBLAS_OP_N, CUBLAS_OP_T, n, n, k, &one, A, CUDA_R_64F, n, A, CUDA_R_64F, n, &zero, dst, CUDA_R_64F, n,
                      CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT));
    };
    auto mirror = [&](double* M) { mirror_lower<<<tg, tb>>>(M, n); };
    auto mperm = [&](const double* src, double* dst) { mirror_permute<<<pg, tb>>>(src, dst, n, cs.r1A); };
    auto fperm = [&](const double* src, double* dst) { permute_full<<<pg, tb>>>(src, dst, n, cs.r1A); };
    auto ctc = [&](double* dst) { CT(cutensorContract(th, plan, &one, A, A, &zero, dst, dst, ctws, ctw, st)); };

    // reference: native dgemm (+ permute)
    if (!r1) gemm(h, Cref); else { gemm(h, T1); fperm(T1, Cref); }
    CK(cudaDeviceSynchronize());
    std::vector<Method> ms;
    if (!r1) {
      ms = {{"dsyrk", false, [&] { syrk(h, C); }, C, 1},
            {"mirror", false, [&] { mirror(C); }, C, 0},
            {"dsyrk+mirror", true, [&] { syrk(h, C); mirror(C); }, C, 0},
            {"dsyrkx+mirror", true, [&] { syrkx(C); mirror(C); }, C, 0},
            {"dgemm", true, [&] { gemm(h, C); }, C, 0},
            {"emu55", true, [&] { emu(C); }, C, 0},
            {"emumath-dgemm", true, [&] { gemm(sh, C); }, C, 0},
            {"emumath-dsyrk+mirror", true, [&] { syrk(sh, C); mirror(C); }, C, 0},
            {"cutensor", true, [&] { ctc(C); }, C, 0}};
    } else {
      ms = {{"dsyrk", false, [&] { syrk(h, G); }, G, 1},
            {"mirror_permute", false, [&] { mperm(G, C); }, C, 0},
            {"dsyrk+mirror_permute", true, [&] { syrk(h, G); mperm(G, C); }, C, 0},
            {"dgemm+permute", true, [&] { gemm(h, G); fperm(G, C); }, C, 0},
            {"permute", false, [&] { fperm(G, C); }, C, 0},
            {"emu55+permute", true, [&] { emu(G); fperm(G, C); }, C, 0},
            {"emumath-dsyrk+mirror_permute", true, [&] { syrk(sh, G); mperm(G, C); }, C, 0},
            {"cutensor", true, [&] { ctc(C); }, C, 0}};
    }
    // checks (outside timing): complete methods against the reference
    std::vector<std::string> chk(ms.size());
    for (size_t q = 0; q < ms.size(); ++q) {
      if (!ms[q].complete) continue;
      CK(cudaMemset(C, 0xFF, nc * 8)); if (G) CK(cudaMemset(G, 0xFF, nc * 8));
      ms[q].run(); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
      const Cmp c = compare(C, Cref, n, false, dres);
      char b[160]; std::snprintf(b, sizeof b, ",\"bits_vs_dgemm\":%llu,\"rel_vs_dgemm\":%.3e", c.bits, c.maxref > 0 ? c.maxdiff / c.maxref : -1.0);
      chk[q] = b;
    }
    // emulation-routing evidence (lower triangle for syrk, full for gemm): T1 = native, T2 = emulated by the math mode
    {
      double* Ce; CK(cudaMalloc(&Ce, nc * 8));
      emu(Ce); syrk(h, T1); syrk(sh, T2); CK(cudaDeviceSynchronize());
      const Cmp s_nat = compare(T2, T1, n, true, dres), s_emu = compare(T2, Ce, n, true, dres), nat_emu = compare(T1, Ce, n, true, dres);
      gemm(h, T1); gemm(sh, T2); CK(cudaDeviceSynchronize());
      const Cmp g_nat = compare(T2, T1, n, false, dres), g_emu = compare(T2, Ce, n, false, dres);
      char line[768];
      std::snprintf(line, sizeof line,
                    "{\"case\":\"%s\",\"check\":\"emulation_routing\",\"emumath_dsyrk_vs_dsyrk_bits\":%llu,\"emumath_dsyrk_vs_dsyrk_rel\":%.3e,"
                    "\"emumath_dsyrk_vs_emu55_bits\":%llu,\"emumath_dsyrk_vs_emu55_rel\":%.3e,\"dsyrk_vs_emu55_rel\":%.3e,"
                    "\"emumath_dgemm_vs_dgemm_bits\":%llu,\"emumath_dgemm_vs_dgemm_rel\":%.3e,\"emumath_dgemm_vs_emu55_bits\":%llu,\"emumath_dgemm_vs_emu55_rel\":%.3e,"
                    "\"lower_elems\":%zu}\n",
                    cs.name.c_str(), s_nat.bits, s_nat.maxdiff / s_nat.maxref, s_emu.bits, s_emu.maxdiff / s_emu.maxref, nat_emu.maxdiff / nat_emu.maxref,
                    g_nat.bits, g_nat.maxdiff / g_nat.maxref, g_emu.bits, g_emu.maxdiff / g_emu.maxref, size_t(n) * (n + 1) / 2);
      emit(line);
      CK(cudaFree(Ce));
    }
    // fill the lower triangle once so that the "mirror" / "mirror_permute" / "permute" methods alone read real data
    if (!r1) syrk(h, C); else gemm(h, G);
    CK(cudaDeviceSynchronize());
    // timing: warm-ups, then interleaved call by call
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int w = 0; w < 3; ++w) for (auto& m : ms) m.run();
    CK(cudaDeviceSynchronize());
    std::vector<std::vector<float>> t(ms.size());
    for (int r = 0; r < reps; ++r)
      for (size_t q = 0; q < ms.size(); ++q) {
        CK(cudaEventRecord(e0, st)); ms[q].run(); CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
        float x; CK(cudaEventElapsedTime(&x, e0, e1)); t[q].push_back(x);
      }
    CK(cudaGetLastError());
    const double fl = 2.0 * n * double(n) * k;
    for (size_t q = 0; q < ms.size(); ++q) {
      auto& v = t[q]; std::sort(v.begin(), v.end()); const size_t z = v.size();
      const double med = z % 2 ? v[z / 2] : 0.5 * (v[z / 2 - 1] + v[z / 2]);
      char line[768];
      std::snprintf(line, sizeof line,
                    "{\"case\":\"%s\",\"n\":%d,\"k\":%d,\"r1_a\":%d,\"method\":\"%s\",\"complete\":%d,\"tflops_eq\":%.2f,\"ms_med\":%.4f,\"ms_min\":%.4f,"
                    "\"ms_p10\":%.4f,\"ms_p90\":%.4f,\"ms_max\":%.4f,\"reps\":%zu%s}\n",
                    cs.name.c_str(), n, k, cs.r1A, ms[q].name.c_str(), int(ms[q].complete), fl / (med * 1e9), med, v.front(), v[z / 10], v[(9 * z) / 10],
                    v.back(), z, chk[q].c_str());
      emit(line);
    }
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    cutensorDestroyPlan(plan); cutensorDestroyPlanPreference(pref); cutensorDestroyOperationDescriptor(op);
    for (auto d : {da, db, dc}) cutensorDestroyTensorDescriptor(d);
    for (void* p : {(void*)A, (void*)C, (void*)Cref, (void*)G, (void*)T1, (void*)T2, ctws}) if (p) CK(cudaFree(p));
  }
  cutensorDestroy(th);
  cublasDestroy(h); cublasDestroy(eh); cublasDestroy(sh);
  if (out) std::fclose(out);
  return 0;
}
