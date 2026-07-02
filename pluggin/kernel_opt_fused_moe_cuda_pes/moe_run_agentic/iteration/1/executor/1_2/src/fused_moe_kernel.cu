#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kGemm1BM = 8;
constexpr int kGemm1BN = 16;
constexpr int kGemm1KSplit = 4;
constexpr int kGemm1Threads = kGemm1BM * kGemm1BN * kGemm1KSplit;

constexpr int kGemm2BM = 8;
constexpr int kGemm2BN = 16;
constexpr int kGemm2KSplit = 4;
constexpr int kGemm2Threads = kGemm2BM * kGemm2BN * kGemm2KSplit;

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float e2m1_to_float(uint8_t code) {
  switch (code & 0x0F) {
    case 0x0:
      return 0.0f;
    case 0x1:
      return 0.5f;
    case 0x2:
      return 1.0f;
    case 0x3:
      return 1.5f;
    case 0x4:
      return 2.0f;
    case 0x5:
      return 3.0f;
    case 0x6:
      return 4.0f;
    case 0x7:
      return 6.0f;
    case 0x8:
      return -0.0f;
    case 0x9:
      return -0.5f;
    case 0xA:
      return -1.0f;
    case 0xB:
      return -1.5f;
    case 0xC:
      return -2.0f;
    case 0xD:
      return -3.0f;
    case 0xE:
      return -4.0f;
    default:
      return -6.0f;
  }
}

__device__ __forceinline__ float load_fp4_scaled(const uint8_t* __restrict__ packed,
                                                 const float* __restrict__ scales,
                                                 int64_t packed_base, int64_t scale_base,
                                                 int col, int scale_vec_size) {
  const uint8_t byte = packed[packed_base + col / 2];
  const uint8_t nibble = (col & 1) == 0 ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
  return e2m1_to_float(nibble) * scales[scale_base + col / scale_vec_size];
}

__global__ void count_local_routes_kernel(const int64_t* __restrict__ topk_ids,
                                          int* __restrict__ route_counts,
                                          int M,
                                          int num_experts,
                                          int local_expert_offset,
                                          int E_local) {
  const int tk = blockIdx.x * blockDim.x + threadIdx.x;
  if (tk >= M) {
    return;
  }

  const int64_t global_expert_raw = topk_ids[tk];
  if (global_expert_raw < 0 || global_expert_raw >= num_experts) {
    return;
  }
  const int le = static_cast<int>(global_expert_raw) - local_expert_offset;
  if (le < 0 || le >= E_local) {
    return;
  }
  atomicAdd(route_counts + le, 1);
}

__global__ void route_offsets_kernel(const int* __restrict__ route_counts,
                                     int* __restrict__ route_offsets,
                                     int* __restrict__ route_cursors,
                                     int E_local) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int running = 0;
  for (int e = 0; e < E_local; ++e) {
    route_offsets[e] = running;
    route_cursors[e] = 0;
    running += route_counts[e];
  }
  route_offsets[E_local] = running;
}

__global__ void scatter_local_routes_kernel(const int64_t* __restrict__ topk_ids,
                                            const int* __restrict__ route_offsets,
                                            int* __restrict__ route_cursors,
                                            int* __restrict__ route_rows,
                                            int M,
                                            int num_experts,
                                            int local_expert_offset,
                                            int E_local) {
  const int tk = blockIdx.x * blockDim.x + threadIdx.x;
  if (tk >= M) {
    return;
  }

  const int64_t global_expert_raw = topk_ids[tk];
  if (global_expert_raw < 0 || global_expert_raw >= num_experts) {
    return;
  }
  const int le = static_cast<int>(global_expert_raw) - local_expert_offset;
  if (le < 0 || le >= E_local) {
    return;
  }

  const int slot = atomicAdd(route_cursors + le, 1);
  route_rows[route_offsets[le] + slot] = tk;
}

__global__ void stage1_grouped_gemm_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const uint8_t* __restrict__ gemm1_weights,
    const float* __restrict__ gemm1_weights_scale,
    const float* __restrict__ gemm1_bias,
    const int64_t* __restrict__ topk_ids,
    const int* __restrict__ route_rows,
    float* __restrict__ mid,
    int M,
    int H,
    int I,
    int E_local,
    int TOPK,
    int num_experts,
    int local_expert_offset,
    int gemm1_scale_cols,
    bool has_gemm1_bias) {
  __shared__ float partial_x1[kGemm1Threads];
  __shared__ float partial_x2[kGemm1Threads];

  const int tid = threadIdx.x;
  const int elem = tid % (kGemm1BM * kGemm1BN);
  const int k_lane = tid / (kGemm1BM * kGemm1BN);
  const int route_slot = blockIdx.y * kGemm1BM + elem / kGemm1BN;
  const int i = blockIdx.x * kGemm1BN + elem % kGemm1BN;

  int tk = -1;
  int t = 0;
  int le = -1;
  bool valid = false;
  if (route_slot < M && i < I) {
    tk = route_rows[route_slot];
    if (tk >= 0) {
      t = tk / TOPK;
      const int64_t global_expert_raw = topk_ids[tk];
      if (global_expert_raw >= 0 && global_expert_raw < num_experts) {
        le = static_cast<int>(global_expert_raw) - local_expert_offset;
        valid = le >= 0 && le < E_local;
      }
    }
  }

  const int gemm1_scale_vec = H / gemm1_scale_cols;
  float sum_x1 = 0.0f;
  float sum_x2 = 0.0f;
  if (valid) {
    const int64_t w1_x1_packed_base =
        ((static_cast<int64_t>(le) * 2 * I + i) * (H / 2));
    const int64_t w1_x2_packed_base =
        ((static_cast<int64_t>(le) * 2 * I + I + i) * (H / 2));
    const int64_t w1_x1_scale_base =
        ((static_cast<int64_t>(le) * 2 * I + i) * gemm1_scale_cols);
    const int64_t w1_x2_scale_base =
        ((static_cast<int64_t>(le) * 2 * I + I + i) * gemm1_scale_cols);
    const int64_t hidden_base = static_cast<int64_t>(t) * H;

    for (int j = k_lane; j < H; j += kGemm1KSplit) {
      const float hidden = __bfloat162float(hidden_states[hidden_base + j]);
      const float w1_x1 = load_fp4_scaled(gemm1_weights, gemm1_weights_scale,
                                          w1_x1_packed_base, w1_x1_scale_base, j,
                                          gemm1_scale_vec);
      const float w1_x2 = load_fp4_scaled(gemm1_weights, gemm1_weights_scale,
                                          w1_x2_packed_base, w1_x2_scale_base, j,
                                          gemm1_scale_vec);
      sum_x1 = fmaf(hidden, w1_x1, sum_x1);
      sum_x2 = fmaf(hidden, w1_x2, sum_x2);
    }
  }

  partial_x1[tid] = sum_x1;
  partial_x2[tid] = sum_x2;
  __syncthreads();

  if (k_lane == 0 && valid) {
    float x1 = has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + i] : 0.0f;
    float x2 =
        has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + I + i] : 0.0f;
    for (int lane = 0; lane < kGemm1KSplit; ++lane) {
      const int offset = lane * (kGemm1BM * kGemm1BN) + elem;
      x1 += partial_x1[offset];
      x2 += partial_x2[offset];
    }
    mid[static_cast<int64_t>(tk) * I + i] = silu(x2) * x1;
  }
}

__global__ void stage2_grouped_gemm_kernel(
    const float* __restrict__ mid,
    const uint8_t* __restrict__ gemm2_weights,
    const float* __restrict__ gemm2_weights_scale,
    const float* __restrict__ gemm2_bias,
    const int64_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const int* __restrict__ route_rows,
    float* __restrict__ out_accum,
    int M,
    int H,
    int I,
    int E_local,
    int TOPK,
    int num_experts,
    int local_expert_offset,
    int gemm2_scale_cols,
    bool has_gemm2_bias) {
  __shared__ float partial[kGemm2Threads];

  const int tid = threadIdx.x;
  const int elem = tid % (kGemm2BM * kGemm2BN);
  const int k_lane = tid / (kGemm2BM * kGemm2BN);
  const int route_slot = blockIdx.y * kGemm2BM + elem / kGemm2BN;
  const int h = blockIdx.x * kGemm2BN + elem % kGemm2BN;

  int tk = -1;
  int t = 0;
  int le = -1;
  bool valid = false;
  if (route_slot < M && h < H) {
    tk = route_rows[route_slot];
    if (tk >= 0) {
      t = tk / TOPK;
      const int64_t global_expert_raw = topk_ids[tk];
      if (global_expert_raw >= 0 && global_expert_raw < num_experts) {
        le = static_cast<int>(global_expert_raw) - local_expert_offset;
        valid = le >= 0 && le < E_local;
      }
    }
  }

  const int gemm2_scale_vec = I / gemm2_scale_cols;
  float sum = 0.0f;
  if (valid) {
    const int64_t w2_packed_base = (static_cast<int64_t>(le) * H + h) * (I / 2);
    const int64_t w2_scale_base = (static_cast<int64_t>(le) * H + h) * gemm2_scale_cols;
    const int64_t mid_base = static_cast<int64_t>(tk) * I;
    for (int i = k_lane; i < I; i += kGemm2KSplit) {
      const float w2 = load_fp4_scaled(gemm2_weights, gemm2_weights_scale, w2_packed_base,
                                       w2_scale_base, i, gemm2_scale_vec);
      sum = fmaf(mid[mid_base + i], w2, sum);
    }
  }

  partial[tid] = sum;
  __syncthreads();

  if (k_lane == 0 && valid) {
    float expert_sum = has_gemm2_bias ? gemm2_bias[static_cast<int64_t>(le) * H + h] : 0.0f;
    for (int lane = 0; lane < kGemm2KSplit; ++lane) {
      expert_sum += partial[lane * (kGemm2BM * kGemm2BN) + elem];
    }
    atomicAdd(out_accum + static_cast<int64_t>(t) * H + h, topk_weights[tk] * expert_sum);
  }
}

__global__ void finalize_output_kernel(const float* __restrict__ out_accum,
                                       __nv_bfloat16* __restrict__ out,
                                       int64_t total) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < total) {
    out[idx] = __float2bfloat16(out_accum[idx]);
  }
}

}  // namespace

void fused_moe_forward_cuda(torch::Tensor hidden_states, torch::Tensor gemm1_weights,
                            torch::Tensor gemm1_weights_scale, torch::Tensor gemm1_bias,
                            torch::Tensor gemm2_weights, torch::Tensor gemm2_weights_scale,
                            torch::Tensor gemm2_bias, torch::Tensor topk_ids,
                            torch::Tensor topk_weights, torch::Tensor out,
                            int64_t num_experts, int64_t local_expert_offset,
                            int64_t intermediate_size) {
  const int T = static_cast<int>(hidden_states.size(0));
  const int H = static_cast<int>(hidden_states.size(1));
  const int I = static_cast<int>(intermediate_size);
  const int E_local = static_cast<int>(gemm1_weights.size(0));
  const int TOPK = static_cast<int>(topk_ids.size(1));
  const int gemm1_scale_cols = static_cast<int>(gemm1_weights_scale.size(2));
  const int gemm2_scale_cols = static_cast<int>(gemm2_weights_scale.size(2));
  const int M = T * TOPK;
  const int64_t out_elems = static_cast<int64_t>(T) * H;

  auto mid = torch::empty({T * TOPK, I}, hidden_states.options().dtype(torch::kFloat32));
  auto out_accum = torch::empty({T, H}, hidden_states.options().dtype(torch::kFloat32));
  auto route_counts = torch::empty({E_local}, topk_ids.options().dtype(torch::kInt32));
  auto route_offsets = torch::empty({E_local + 1}, topk_ids.options().dtype(torch::kInt32));
  auto route_cursors = torch::empty({E_local}, topk_ids.options().dtype(torch::kInt32));
  auto route_rows = torch::empty({M}, topk_ids.options().dtype(torch::kInt32));
  const auto stream = at::cuda::getCurrentCUDAStream();

  if (E_local > 0) {
    C10_CUDA_CHECK(cudaMemsetAsync(route_counts.data_ptr<int>(), 0,
                                   static_cast<size_t>(E_local) * sizeof(int), stream));
    C10_CUDA_CHECK(cudaMemsetAsync(route_cursors.data_ptr<int>(), 0,
                                   static_cast<size_t>(E_local) * sizeof(int), stream));
  }
  if (M > 0) {
    C10_CUDA_CHECK(cudaMemsetAsync(route_rows.data_ptr<int>(), 0xFF,
                                   static_cast<size_t>(M) * sizeof(int), stream));
  }
  if (out_elems > 0) {
    C10_CUDA_CHECK(cudaMemsetAsync(out_accum.data_ptr<float>(), 0,
                                   static_cast<size_t>(out_elems) * sizeof(float), stream));
  }

  if (M > 0 && E_local > 0) {
    constexpr int route_threads = 128;
    const int route_blocks = (M + route_threads - 1) / route_threads;
    count_local_routes_kernel<<<route_blocks, route_threads, 0, stream>>>(
        topk_ids.data_ptr<int64_t>(), route_counts.data_ptr<int>(), M,
        static_cast<int>(num_experts), static_cast<int>(local_expert_offset), E_local);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  route_offsets_kernel<<<1, 1, 0, stream>>>(route_counts.data_ptr<int>(),
                                            route_offsets.data_ptr<int>(),
                                            route_cursors.data_ptr<int>(), E_local);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (M > 0 && E_local > 0) {
    constexpr int route_threads = 128;
    const int route_blocks = (M + route_threads - 1) / route_threads;
    scatter_local_routes_kernel<<<route_blocks, route_threads, 0, stream>>>(
        topk_ids.data_ptr<int64_t>(), route_offsets.data_ptr<int>(),
        route_cursors.data_ptr<int>(), route_rows.data_ptr<int>(), M,
        static_cast<int>(num_experts), static_cast<int>(local_expert_offset), E_local);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (M > 0 && I > 0) {
    constexpr dim3 block_stage1(kGemm1Threads);
    const dim3 grid_stage1((I + kGemm1BN - 1) / kGemm1BN, (M + kGemm1BM - 1) / kGemm1BM);
    stage1_grouped_gemm_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
        gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
        gemm1_bias.data_ptr<float>(), topk_ids.data_ptr<int64_t>(), route_rows.data_ptr<int>(),
        mid.data_ptr<float>(), M, H, I, E_local, TOPK, static_cast<int>(num_experts),
        static_cast<int>(local_expert_offset), gemm1_scale_cols, gemm1_bias.numel() > 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (M > 0 && H > 0) {
    constexpr dim3 block_stage2(kGemm2Threads);
    const dim3 grid_stage2((H + kGemm2BN - 1) / kGemm2BN, (M + kGemm2BM - 1) / kGemm2BM);
    stage2_grouped_gemm_kernel<<<grid_stage2, block_stage2, 0, stream>>>(
        mid.data_ptr<float>(), gemm2_weights.data_ptr<uint8_t>(),
        gemm2_weights_scale.data_ptr<float>(), gemm2_bias.data_ptr<float>(),
        topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(), route_rows.data_ptr<int>(),
        out_accum.data_ptr<float>(), M, H, I, E_local, TOPK, static_cast<int>(num_experts),
        static_cast<int>(local_expert_offset), gemm2_scale_cols, gemm2_bias.numel() > 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (out_elems > 0) {
    constexpr int finalize_threads = 256;
    const int finalize_blocks =
        static_cast<int>((out_elems + finalize_threads - 1) / finalize_threads);
    finalize_output_kernel<<<finalize_blocks, finalize_threads, 0, stream>>>(
        out_accum.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), out_elems);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
