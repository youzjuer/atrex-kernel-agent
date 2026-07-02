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

__device__ __constant__ float kE2M1Lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float decode_fp4(uint32_t code) {
  return kE2M1Lut[code & 0x0F];
}

__device__ __forceinline__ bool is_u32_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0;
}

__device__ __forceinline__ int first_k_split_index(int begin, int k_lane, int k_split) {
  const int rem = begin % k_split;
  int delta = k_lane - rem;
  if (delta < 0) {
    delta += k_split;
  }
  return begin + delta;
}

__device__ __forceinline__ float load_fp4_unscaled(const uint8_t* __restrict__ row, int col) {
  const uint8_t byte = row[col >> 1];
  return decode_fp4(byte >> (4 * (col & 1)));
}

__device__ __forceinline__ void accumulate_fp4_pair_scaled_ksplit_tile(
    const __nv_bfloat16* __restrict__ hidden,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ row2,
    int begin,
    int end,
    int k_lane,
    int k_split,
    float scale1,
    float scale2,
    float& acc1,
    float& acc2) {
  int j = first_k_split_index(begin, k_lane, k_split);
  if (j >= end) {
    return;
  }

  if (k_split == 4) {
    while (j < end && ((j & 7) >= 4 || (j & ~7) < begin)) {
      const float hidden_val = __bfloat162float(hidden[j]);
      acc1 = fmaf(hidden_val, load_fp4_unscaled(row1, j) * scale1, acc1);
      acc2 = fmaf(hidden_val, load_fp4_unscaled(row2, j) * scale2, acc2);
      j += 4;
    }

    while (j + 4 < end && ((j & ~7) + 8) <= end) {
      const int word_base = j & ~7;
      const uint8_t* ptr1 = row1 + (word_base >> 1);
      const uint8_t* ptr2 = row2 + (word_base >> 1);
      if (is_u32_aligned(ptr1) && is_u32_aligned(ptr2)) {
        const uint32_t packed1 = __ldg(reinterpret_cast<const uint32_t*>(ptr1));
        const uint32_t packed2 = __ldg(reinterpret_cast<const uint32_t*>(ptr2));
        const int shift = 4 * (j - word_base);
        const float hidden0 = __bfloat162float(hidden[j]);
        const float hidden1 = __bfloat162float(hidden[j + 4]);
        acc1 = fmaf(hidden0, decode_fp4(packed1 >> shift) * scale1, acc1);
        acc2 = fmaf(hidden0, decode_fp4(packed2 >> shift) * scale2, acc2);
        acc1 = fmaf(hidden1, decode_fp4(packed1 >> (shift + 16)) * scale1, acc1);
        acc2 = fmaf(hidden1, decode_fp4(packed2 >> (shift + 16)) * scale2, acc2);
      } else {
        const float hidden0 = __bfloat162float(hidden[j]);
        const float hidden1 = __bfloat162float(hidden[j + 4]);
        acc1 = fmaf(hidden0, load_fp4_unscaled(row1, j) * scale1, acc1);
        acc2 = fmaf(hidden0, load_fp4_unscaled(row2, j) * scale2, acc2);
        acc1 = fmaf(hidden1, load_fp4_unscaled(row1, j + 4) * scale1, acc1);
        acc2 = fmaf(hidden1, load_fp4_unscaled(row2, j + 4) * scale2, acc2);
      }
      j += 8;
    }
  }

  while (j < end) {
    const float hidden_val = __bfloat162float(hidden[j]);
    acc1 = fmaf(hidden_val, load_fp4_unscaled(row1, j) * scale1, acc1);
    acc2 = fmaf(hidden_val, load_fp4_unscaled(row2, j) * scale2, acc2);
    j += k_split;
  }
}

__device__ __forceinline__ void accumulate_fp4_scaled_ksplit_tile(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ row,
    int begin,
    int end,
    int k_lane,
    int k_split,
    float scale,
    float& acc) {
  int i = first_k_split_index(begin, k_lane, k_split);
  if (i >= end) {
    return;
  }

  if (k_split == 4) {
    while (i < end && ((i & 7) >= 4 || (i & ~7) < begin)) {
      acc = fmaf(activations[i], load_fp4_unscaled(row, i) * scale, acc);
      i += 4;
    }

    while (i + 4 < end && ((i & ~7) + 8) <= end) {
      const int word_base = i & ~7;
      const uint8_t* ptr = row + (word_base >> 1);
      if (is_u32_aligned(ptr)) {
        const uint32_t packed = __ldg(reinterpret_cast<const uint32_t*>(ptr));
        const int shift = 4 * (i - word_base);
        acc = fmaf(activations[i], decode_fp4(packed >> shift) * scale, acc);
        acc = fmaf(activations[i + 4], decode_fp4(packed >> (shift + 16)) * scale, acc);
      } else {
        acc = fmaf(activations[i], load_fp4_unscaled(row, i) * scale, acc);
        acc = fmaf(activations[i + 4], load_fp4_unscaled(row, i + 4) * scale, acc);
      }
      i += 8;
    }
  }

  while (i < end) {
    acc = fmaf(activations[i], load_fp4_unscaled(row, i) * scale, acc);
    i += k_split;
  }
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
    const __nv_bfloat16* hidden_row = hidden_states + hidden_base;
    const uint8_t* w1_x1_row = gemm1_weights + w1_x1_packed_base;
    const uint8_t* w1_x2_row = gemm1_weights + w1_x2_packed_base;

    for (int scale_col = 0; scale_col < gemm1_scale_cols; ++scale_col) {
      const int begin = scale_col * gemm1_scale_vec;
      const int end = begin + gemm1_scale_vec;
      const float scale1 = __ldg(gemm1_weights_scale + w1_x1_scale_base + scale_col);
      const float scale2 = __ldg(gemm1_weights_scale + w1_x2_scale_base + scale_col);
      accumulate_fp4_pair_scaled_ksplit_tile(hidden_row, w1_x1_row, w1_x2_row, begin, end,
                                             k_lane, kGemm1KSplit, scale1, scale2, sum_x1,
                                             sum_x2);
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
    const float* mid_row = mid + mid_base;
    const uint8_t* w2_row = gemm2_weights + w2_packed_base;
    for (int scale_col = 0; scale_col < gemm2_scale_cols; ++scale_col) {
      const int begin = scale_col * gemm2_scale_vec;
      const int end = begin + gemm2_scale_vec;
      const float scale = __ldg(gemm2_weights_scale + w2_scale_base + scale_col);
      accumulate_fp4_scaled_ksplit_tile(mid_row, w2_row, begin, end, k_lane, kGemm2KSplit,
                                        scale, sum);
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
