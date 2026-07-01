#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

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

__global__ void build_grouped_routes_kernel(
    const int64_t* __restrict__ topk_ids,
    int* __restrict__ expert_counts,
    int* __restrict__ expert_offsets,
    int* __restrict__ grouped_route_ids,
    int* __restrict__ route_to_grouped,
    int total_routes,
    int E_local,
    int num_experts,
    int local_expert_offset) {
  extern __shared__ int smem[];
  int* counts = smem;
  int* cursors = smem + E_local;
  const int tid = threadIdx.x;

  for (int e = tid; e < E_local; e += blockDim.x) {
    counts[e] = 0;
  }
  for (int r = tid; r < total_routes; r += blockDim.x) {
    grouped_route_ids[r] = -1;
    route_to_grouped[r] = -1;
  }
  __syncthreads();

  for (int r = tid; r < total_routes; r += blockDim.x) {
    const int64_t global_expert_raw = topk_ids[r];
    if (global_expert_raw < 0 || global_expert_raw >= num_experts) {
      continue;
    }
    const int le = static_cast<int>(global_expert_raw) - local_expert_offset;
    if (le >= 0 && le < E_local) {
      atomicAdd(&counts[le], 1);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int total = 0;
    expert_offsets[0] = 0;
    for (int e = 0; e < E_local; ++e) {
      const int count = counts[e];
      expert_counts[e] = count;
      cursors[e] = total;
      total += count;
      expert_offsets[e + 1] = total;
    }
  }
  __syncthreads();

  for (int r = tid; r < total_routes; r += blockDim.x) {
    const int64_t global_expert_raw = topk_ids[r];
    if (global_expert_raw < 0 || global_expert_raw >= num_experts) {
      continue;
    }
    const int le = static_cast<int>(global_expert_raw) - local_expert_offset;
    if (le < 0 || le >= E_local) {
      continue;
    }
    const int grouped_idx = atomicAdd(&cursors[le], 1);
    grouped_route_ids[grouped_idx] = r;
    route_to_grouped[r] = grouped_idx;
  }
}

__global__ void stage1_activation_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const uint8_t* __restrict__ gemm1_weights,
    const float* __restrict__ gemm1_weights_scale,
    const float* __restrict__ gemm1_bias,
    const int64_t* __restrict__ topk_ids,
    const int* __restrict__ grouped_route_ids,
    float* __restrict__ mid,
    int T,
    int H,
    int I,
    int E_local,
    int TOPK,
    int num_experts,
    int local_expert_offset,
    int gemm1_scale_cols,
    bool has_gemm1_bias) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int grouped_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= I || grouped_idx >= T * TOPK) {
    return;
  }

  const int route = grouped_route_ids[grouped_idx];
  if (route < 0) {
    return;
  }

  const int t = route / TOPK;
  const int64_t global_expert_raw = topk_ids[route];
  if (global_expert_raw < 0 || global_expert_raw >= num_experts) {
    return;
  }
  const int le = static_cast<int>(global_expert_raw) - local_expert_offset;
  if (le < 0 || le >= E_local) {
    return;
  }

  const int gemm1_scale_vec = H / gemm1_scale_cols;
  float x1 = has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + i] : 0.0f;
  float x2 = has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + I + i] : 0.0f;
  const int64_t w1_x1_packed_base = ((static_cast<int64_t>(le) * 2 * I + i) * (H / 2));
  const int64_t w1_x2_packed_base =
      ((static_cast<int64_t>(le) * 2 * I + I + i) * (H / 2));
  const int64_t w1_x1_scale_base =
      ((static_cast<int64_t>(le) * 2 * I + i) * gemm1_scale_cols);
  const int64_t w1_x2_scale_base =
      ((static_cast<int64_t>(le) * 2 * I + I + i) * gemm1_scale_cols);
  const int64_t hidden_base = static_cast<int64_t>(t) * H;

  for (int j = 0; j < H; ++j) {
    const float hidden = __bfloat162float(hidden_states[hidden_base + j]);
    const float w1_x1 = load_fp4_scaled(gemm1_weights, gemm1_weights_scale,
                                        w1_x1_packed_base, w1_x1_scale_base, j,
                                        gemm1_scale_vec);
    const float w1_x2 = load_fp4_scaled(gemm1_weights, gemm1_weights_scale,
                                        w1_x2_packed_base, w1_x2_scale_base, j,
                                        gemm1_scale_vec);
    x1 = fmaf(hidden, w1_x1, x1);
    x2 = fmaf(hidden, w1_x2, x2);
  }
  mid[static_cast<int64_t>(grouped_idx) * I + i] = silu(x2) * x1;
}

__global__ void stage2_grouped_output_kernel(
    const float* __restrict__ mid,
    const uint8_t* __restrict__ gemm2_weights,
    const float* __restrict__ gemm2_weights_scale,
    const float* __restrict__ gemm2_bias,
    const int64_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const int* __restrict__ grouped_route_ids,
    const int* __restrict__ route_to_grouped,
    float* __restrict__ out_accum,
    int T,
    int H,
    int I,
    int E_local,
    int TOPK,
    int num_experts,
    int local_expert_offset,
    int gemm2_scale_cols,
    bool has_gemm2_bias) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int grouped_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (grouped_idx >= T * TOPK || h >= H) {
    return;
  }

  const int route = grouped_route_ids[grouped_idx];
  if (route < 0) {
    return;
  }
  if (route_to_grouped[route] != grouped_idx) {
    return;
  }
  const int t = route / TOPK;
  const int k = route - t * TOPK;
  if (t >= T) {
    return;
  }
  const int64_t global_expert_raw = topk_ids[route];
  if (global_expert_raw < 0 || global_expert_raw >= num_experts) {
    return;
  }
  const int le = static_cast<int>(global_expert_raw) - local_expert_offset;
  if (le < 0 || le >= E_local) {
    return;
  }

  const int gemm2_scale_vec = I / gemm2_scale_cols;
  float expert_sum = 0.0f;
  const int64_t w2_packed_base = (static_cast<int64_t>(le) * H + h) * (I / 2);
  const int64_t w2_scale_base = (static_cast<int64_t>(le) * H + h) * gemm2_scale_cols;
  for (int i = 0; i < I; ++i) {
    const float w2 = load_fp4_scaled(gemm2_weights, gemm2_weights_scale, w2_packed_base,
                                     w2_scale_base, i, gemm2_scale_vec);
    expert_sum = fmaf(mid[static_cast<int64_t>(grouped_idx) * I + i], w2, expert_sum);
  }
  if (has_gemm2_bias) {
    expert_sum += gemm2_bias[static_cast<int64_t>(le) * H + h];
  }

  atomicAdd(&out_accum[static_cast<int64_t>(t) * H + h],
            topk_weights[static_cast<int64_t>(t) * TOPK + k] * expert_sum);
}

__global__ void finalize_output_kernel(const float* __restrict__ out_accum,
                                       __nv_bfloat16* __restrict__ out,
                                       int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
  const int total_routes = T * TOPK;

  auto int_options = hidden_states.options().dtype(torch::kInt32);
  auto expert_counts = torch::empty({E_local}, int_options);
  auto expert_offsets = torch::empty({E_local + 1}, int_options);
  auto grouped_route_ids = torch::empty({total_routes}, int_options);
  auto route_to_grouped = torch::empty({total_routes}, int_options);
  auto mid = torch::empty({total_routes, I}, hidden_states.options().dtype(torch::kFloat32));
  auto out_accum = torch::empty({T, H}, hidden_states.options().dtype(torch::kFloat32));
  const auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int scheduling_threads = 256;
  const size_t scheduling_smem = static_cast<size_t>(2 * E_local) * sizeof(int);
  build_grouped_routes_kernel<<<1, scheduling_threads, scheduling_smem, stream>>>(
      topk_ids.data_ptr<int64_t>(), expert_counts.data_ptr<int>(),
      expert_offsets.data_ptr<int>(), grouped_route_ids.data_ptr<int>(),
      route_to_grouped.data_ptr<int>(), total_routes, E_local, static_cast<int>(num_experts),
      static_cast<int>(local_expert_offset));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  C10_CUDA_CHECK(cudaMemsetAsync(out_accum.data_ptr<float>(), 0,
                                 static_cast<size_t>(T) * H * sizeof(float), stream));

  constexpr dim3 block_stage1(16, 4);
  const dim3 grid_stage1((I + block_stage1.x - 1) / block_stage1.x,
                         (total_routes + block_stage1.y - 1) / block_stage1.y);
  stage1_activation_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
      gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
      gemm1_bias.data_ptr<float>(), topk_ids.data_ptr<int64_t>(),
      grouped_route_ids.data_ptr<int>(), mid.data_ptr<float>(), T, H, I, E_local, TOPK,
      static_cast<int>(num_experts), static_cast<int>(local_expert_offset), gemm1_scale_cols,
      gemm1_bias.numel() > 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr dim3 block_stage2(16, 8);
  const dim3 grid_stage2((H + block_stage2.x - 1) / block_stage2.x,
                         (total_routes + block_stage2.y - 1) / block_stage2.y);
  stage2_grouped_output_kernel<<<grid_stage2, block_stage2, 0, stream>>>(
      mid.data_ptr<float>(), gemm2_weights.data_ptr<uint8_t>(),
      gemm2_weights_scale.data_ptr<float>(), gemm2_bias.data_ptr<float>(),
      topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
      grouped_route_ids.data_ptr<int>(), route_to_grouped.data_ptr<int>(),
      out_accum.data_ptr<float>(), T, H, I, E_local, TOPK, static_cast<int>(num_experts),
      static_cast<int>(local_expert_offset), gemm2_scale_cols, gemm2_bias.numel() > 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int finalize_threads = 256;
  const int output_elems = T * H;
  const dim3 grid_finalize((output_elems + finalize_threads - 1) / finalize_threads);
  finalize_output_kernel<<<grid_finalize, finalize_threads, 0, stream>>>(
      out_accum.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      output_elems);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
