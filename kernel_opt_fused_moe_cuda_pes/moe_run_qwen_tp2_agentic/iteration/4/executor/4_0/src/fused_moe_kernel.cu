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

__device__ __constant__ float kE2M1Lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float decode_fp4(uint32_t code) {
  return kE2M1Lut[code & 0x0F];
}

__device__ __forceinline__ bool is_u32_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0;
}

__device__ __forceinline__ void accumulate_fp4_pair_scaled_tile(
    const __nv_bfloat16* __restrict__ hidden,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ row2,
    int begin,
    int end,
    float scale1,
    float scale2,
    float& acc1,
    float& acc2) {
  int j = begin;
  if (j < end && (j & 1)) {
    const uint8_t byte1 = row1[j >> 1];
    const uint8_t byte2 = row2[j >> 1];
    const float hidden_val = __bfloat162float(hidden[j]);
    acc1 = fmaf(hidden_val, decode_fp4(byte1 >> 4) * scale1, acc1);
    acc2 = fmaf(hidden_val, decode_fp4(byte2 >> 4) * scale2, acc2);
    ++j;
  }

  const uint8_t* ptr1 = row1 + (j >> 1);
  const uint8_t* ptr2 = row2 + (j >> 1);
  if (is_u32_aligned(ptr1) && is_u32_aligned(ptr2)) {
    while (j + 8 <= end) {
      const uint32_t packed1 = __ldg(reinterpret_cast<const uint32_t*>(ptr1));
      const uint32_t packed2 = __ldg(reinterpret_cast<const uint32_t*>(ptr2));
#pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        const float hidden_val = __bfloat162float(hidden[j + lane]);
        acc1 = fmaf(hidden_val, decode_fp4(packed1 >> (4 * lane)) * scale1, acc1);
        acc2 = fmaf(hidden_val, decode_fp4(packed2 >> (4 * lane)) * scale2, acc2);
      }
      j += 8;
      ptr1 += 4;
      ptr2 += 4;
    }
  }

  while (j + 2 <= end) {
    const uint8_t byte1 = *ptr1;
    const uint8_t byte2 = *ptr2;
    const float hidden0 = __bfloat162float(hidden[j]);
    const float hidden1 = __bfloat162float(hidden[j + 1]);
    acc1 = fmaf(hidden0, decode_fp4(byte1) * scale1, acc1);
    acc2 = fmaf(hidden0, decode_fp4(byte2) * scale2, acc2);
    acc1 = fmaf(hidden1, decode_fp4(byte1 >> 4) * scale1, acc1);
    acc2 = fmaf(hidden1, decode_fp4(byte2 >> 4) * scale2, acc2);
    j += 2;
    ++ptr1;
    ++ptr2;
  }

  if (j < end) {
    const uint8_t byte1 = *ptr1;
    const uint8_t byte2 = *ptr2;
    const float hidden_val = __bfloat162float(hidden[j]);
    acc1 = fmaf(hidden_val, decode_fp4(byte1) * scale1, acc1);
    acc2 = fmaf(hidden_val, decode_fp4(byte2) * scale2, acc2);
  }
}

__device__ __forceinline__ void accumulate_fp4_pair_scaled_shared_tile(
    const __nv_bfloat16* __restrict__ hidden_tile,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ row2,
    int begin,
    int tile_len,
    float scale1,
    float scale2,
    float& acc1,
    float& acc2) {
  for (int kk = 0; kk < tile_len; ++kk) {
    const int j = begin + kk;
    const uint8_t byte1 = row1[j >> 1];
    const uint8_t byte2 = row2[j >> 1];
    const int shift = (j & 1) ? 4 : 0;
    const float hidden_val = __bfloat162float(hidden_tile[kk]);
    acc1 = fmaf(hidden_val, decode_fp4(byte1 >> shift) * scale1, acc1);
    acc2 = fmaf(hidden_val, decode_fp4(byte2 >> shift) * scale2, acc2);
  }
}

__device__ __forceinline__ void accumulate_fp4_scaled_tile(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ row,
    int begin,
    int end,
    float scale,
    float& acc) {
  int i = begin;
  if (i < end && (i & 1)) {
    const uint8_t byte = row[i >> 1];
    acc = fmaf(activations[i], decode_fp4(byte >> 4) * scale, acc);
    ++i;
  }

  const uint8_t* ptr = row + (i >> 1);
  if (is_u32_aligned(ptr)) {
    while (i + 8 <= end) {
      const uint32_t packed = __ldg(reinterpret_cast<const uint32_t*>(ptr));
#pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        acc = fmaf(activations[i + lane], decode_fp4(packed >> (4 * lane)) * scale, acc);
      }
      i += 8;
      ptr += 4;
    }
  }

  while (i + 2 <= end) {
    const uint8_t byte = *ptr;
    acc = fmaf(activations[i], decode_fp4(byte) * scale, acc);
    acc = fmaf(activations[i + 1], decode_fp4(byte >> 4) * scale, acc);
    i += 2;
    ++ptr;
  }

  if (i < end) {
    acc = fmaf(activations[i], decode_fp4(*ptr) * scale, acc);
  }
}

__global__ void stage1_grouped_activation_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const uint8_t* __restrict__ gemm1_weights,
    const float* __restrict__ gemm1_weights_scale,
    const float* __restrict__ gemm1_bias,
    const int* __restrict__ topk_slots,
    const int* __restrict__ row_to_expert,
    const int* __restrict__ grouped_rows_count,
    const int* __restrict__ a_map,
    float* __restrict__ mid,
    int H,
    int I,
    int E_local,
    int gemm1_scale_cols,
    bool has_gemm1_bias) {
  extern __shared__ unsigned char smem_raw[];
  int* row_slots = reinterpret_cast<int*>(smem_raw);
  int* row_tokens = row_slots + blockDim.y;
  int* row_experts = row_tokens + blockDim.y;
  __nv_bfloat16* sh_hidden = reinterpret_cast<__nv_bfloat16*>(row_experts + blockDim.y);

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_in_block = threadIdx.y;
  const int grouped_idx = blockIdx.y * blockDim.y + row_in_block;
  const int gemm1_scale_vec = H / gemm1_scale_cols;

  if (threadIdx.x == 0) {
    int original_slot = -1;
    int token_idx = -1;
    int local_expert = -1;
    if (grouped_idx < grouped_rows_count[0]) {
      original_slot = topk_slots[grouped_idx];
      token_idx = a_map[grouped_idx];
      local_expert = row_to_expert[grouped_idx];
    }
    row_slots[row_in_block] = original_slot;
    row_tokens[row_in_block] = token_idx;
    row_experts[row_in_block] = local_expert;
  }
  __syncthreads();

  const int original_slot = row_slots[row_in_block];
  const int token_idx = row_tokens[row_in_block];
  const int le = row_experts[row_in_block];
  const bool valid_row = original_slot >= 0 && token_idx >= 0 && le >= 0 && le < E_local;
  const bool valid_output = valid_row && i < I;
  int64_t expert_row_base = 0;
  int64_t w1_x1_scale_base = 0;
  int64_t w1_x2_scale_base = 0;
  const uint8_t* w1_x1_row = gemm1_weights;
  const uint8_t* w1_x2_row = gemm1_weights;
  __nv_bfloat16* hidden_tile = sh_hidden + row_in_block * gemm1_scale_vec;

  float x1 = 0.0f;
  float x2 = 0.0f;
  if (valid_output) {
    expert_row_base = static_cast<int64_t>(le) * 2 * I;
    const int64_t w1_x1_packed_base = (expert_row_base + i) * (H / 2);
    const int64_t w1_x2_packed_base = (expert_row_base + I + i) * (H / 2);
    w1_x1_scale_base = (expert_row_base + i) * gemm1_scale_cols;
    w1_x2_scale_base = (expert_row_base + I + i) * gemm1_scale_cols;
    w1_x1_row = gemm1_weights + w1_x1_packed_base;
    w1_x2_row = gemm1_weights + w1_x2_packed_base;
    if (has_gemm1_bias) {
      x1 = gemm1_bias[expert_row_base + i];
      x2 = gemm1_bias[expert_row_base + I + i];
    }
  }

  for (int scale_col = 0; scale_col < gemm1_scale_cols; ++scale_col) {
    const int begin = scale_col * gemm1_scale_vec;
    const int remaining = H - begin;
    const int tile_len = remaining < gemm1_scale_vec ? remaining : gemm1_scale_vec;
    for (int kk = threadIdx.x; kk < tile_len; kk += blockDim.x) {
      hidden_tile[kk] =
          valid_row ? hidden_states[static_cast<int64_t>(token_idx) * H + begin + kk]
                    : __float2bfloat16(0.0f);
    }
    __syncthreads();

    if (valid_output) {
      const float scale1 = gemm1_weights_scale[w1_x1_scale_base + scale_col];
      const float scale2 = gemm1_weights_scale[w1_x2_scale_base + scale_col];
      accumulate_fp4_pair_scaled_shared_tile(hidden_tile, w1_x1_row, w1_x2_row, begin,
                                             tile_len, scale1, scale2, x1, x2);
    }
    __syncthreads();
  }

  if (valid_output) {
    mid[static_cast<int64_t>(original_slot) * I + i] = silu(x2) * x1;
  }
}

__global__ void build_grouped_expert_schedule_kernel(
    const int64_t* __restrict__ topk_ids,
    int* __restrict__ topk_slots,
    int* __restrict__ expert_counts,
    int* __restrict__ expert_offsets,
    int* __restrict__ a_map,
    int* __restrict__ c_map,
    int* __restrict__ row_to_expert,
    int* __restrict__ grouped_rows_count,
    int* __restrict__ tile_idx_to_expert_idx,
    int* __restrict__ problem_sizes_mnkl,
    int total_routes,
    int TOPK,
    int E_local,
    int num_experts,
    int local_expert_offset,
    int H,
    int I) {
  extern __shared__ int smem[];
  int* counts = smem;
  int* cursors = smem + E_local;
  const int tid = threadIdx.x;

  for (int e = tid; e < E_local; e += blockDim.x) {
    counts[e] = 0;
  }
  for (int r = tid; r < total_routes; r += blockDim.x) {
    topk_slots[r] = -1;
    a_map[r] = -1;
    c_map[r] = -1;
    row_to_expert[r] = -1;
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
    int grouped_rows = 0;
    expert_offsets[0] = 0;
    for (int e = 0; e < E_local; ++e) {
      const int count = counts[e];
      expert_counts[e] = count;
      cursors[e] = grouped_rows;
      grouped_rows += count;
      expert_offsets[e + 1] = grouped_rows;
      tile_idx_to_expert_idx[e] = count > 0 ? e : -1;
      problem_sizes_mnkl[4 * e + 0] = count;
      problem_sizes_mnkl[4 * e + 1] = 2 * I;
      problem_sizes_mnkl[4 * e + 2] = H;
      problem_sizes_mnkl[4 * e + 3] = 1;
    }
    grouped_rows_count[0] = grouped_rows;
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
    topk_slots[grouped_idx] = r;
    a_map[grouped_idx] = r / TOPK;
    c_map[r] = grouped_idx;
    row_to_expert[grouped_idx] = le;
  }
}

__global__ void stage1_activation_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const uint8_t* __restrict__ gemm1_weights,
    const float* __restrict__ gemm1_weights_scale,
    const float* __restrict__ gemm1_bias,
    const int64_t* __restrict__ topk_ids,
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
  const int tk = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= I || tk >= T * TOPK) {
    return;
  }

  const int t = tk / TOPK;
  const int k = tk - t * TOPK;
  const int64_t global_expert_raw = topk_ids[t * TOPK + k];
  if (global_expert_raw < 0 || global_expert_raw >= num_experts) {
    mid[static_cast<int64_t>(tk) * I + i] = 0.0f;
    return;
  }
  const int le = static_cast<int>(global_expert_raw) - local_expert_offset;
  if (le < 0 || le >= E_local) {
    mid[static_cast<int64_t>(tk) * I + i] = 0.0f;
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
  const __nv_bfloat16* hidden_row = hidden_states + static_cast<int64_t>(t) * H;
  const uint8_t* w1_x1_row = gemm1_weights + w1_x1_packed_base;
  const uint8_t* w1_x2_row = gemm1_weights + w1_x2_packed_base;

  for (int scale_col = 0; scale_col < gemm1_scale_cols; ++scale_col) {
    const int begin = scale_col * gemm1_scale_vec;
    const int end = begin + gemm1_scale_vec;
    const float scale1 = gemm1_weights_scale[w1_x1_scale_base + scale_col];
    const float scale2 = gemm1_weights_scale[w1_x2_scale_base + scale_col];
    accumulate_fp4_pair_scaled_tile(hidden_row, w1_x1_row, w1_x2_row, begin, end, scale1,
                                    scale2, x1, x2);
  }
  mid[static_cast<int64_t>(tk) * I + i] = silu(x2) * x1;
}

__global__ void stage2_grouped_down_kernel(
    const float* __restrict__ mid,
    const uint8_t* __restrict__ gemm2_weights,
    const float* __restrict__ gemm2_weights_scale,
    const float* __restrict__ gemm2_bias,
    const int64_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    float* __restrict__ out_accum,
    int T,
    int H,
    int I,
    int E_local,
    int TOPK,
    int num_experts,
    int local_expert_offset,
    int gemm2_scale_cols,
    bool has_gemm2_bias,
    int split_count) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int tk = blockIdx.y * blockDim.y + threadIdx.y;
  const int split = blockIdx.z;
  if (tk >= T * TOPK || h >= H || split >= split_count) {
    return;
  }

  const int t = tk / TOPK;
  const int64_t global_expert_raw = topk_ids[tk];
  if (global_expert_raw < 0 || global_expert_raw >= num_experts) {
    return;
  }
  const int le = static_cast<int>(global_expert_raw) - local_expert_offset;
  if (le < 0 || le >= E_local) {
    return;
  }

  const int gemm2_scale_vec = I / gemm2_scale_cols;
  const int scale_begin = (gemm2_scale_cols * split) / split_count;
  const int scale_end = (gemm2_scale_cols * (split + 1)) / split_count;
  float partial = 0.0f;
  const int64_t w2_packed_base = (static_cast<int64_t>(le) * H + h) * (I / 2);
  const int64_t w2_scale_base = (static_cast<int64_t>(le) * H + h) * gemm2_scale_cols;
  const float* mid_row = mid + static_cast<int64_t>(tk) * I;
  const uint8_t* w2_row = gemm2_weights + w2_packed_base;
  for (int scale_col = scale_begin; scale_col < scale_end; ++scale_col) {
    const int begin = scale_col * gemm2_scale_vec;
    const int end = begin + gemm2_scale_vec;
    const float scale = gemm2_weights_scale[w2_scale_base + scale_col];
    accumulate_fp4_scaled_tile(mid_row, w2_row, begin, end, scale, partial);
  }
  if (split == 0 && has_gemm2_bias) {
    partial += gemm2_bias[static_cast<int64_t>(le) * H + h];
  }
  atomicAdd(out_accum + static_cast<int64_t>(t) * H + h, topk_weights[tk] * partial);
}

__global__ void finalize_output_kernel(
    const float* __restrict__ out_accum,
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

  auto mid = torch::empty({T * TOPK, I}, hidden_states.options().dtype(torch::kFloat32));
  const auto stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor topk_slots;
  torch::Tensor expert_counts;
  torch::Tensor expert_offsets;
  torch::Tensor a_map;
  torch::Tensor c_map;
  torch::Tensor row_to_expert;
  torch::Tensor grouped_rows_count;
  torch::Tensor tile_idx_to_expert_idx;
  torch::Tensor problem_sizes_mnkl;

  bool has_grouped_schedule = false;
  constexpr int kMaxSingleCtaRoutes = 32768;
  if (total_routes <= kMaxSingleCtaRoutes) {
    auto int_options = hidden_states.options().dtype(torch::kInt32);
    topk_slots = torch::empty({total_routes}, int_options);
    expert_counts = torch::empty({E_local}, int_options);
    expert_offsets = torch::empty({E_local + 1}, int_options);
    a_map = torch::empty({total_routes}, int_options);
    c_map = torch::empty({total_routes}, int_options);
    row_to_expert = torch::empty({total_routes}, int_options);
    grouped_rows_count = torch::empty({1}, int_options);
    tile_idx_to_expert_idx = torch::empty({E_local}, int_options);
    problem_sizes_mnkl = torch::empty({E_local, 4}, int_options);

    constexpr int schedule_threads = 256;
    const size_t schedule_smem = static_cast<size_t>(2 * E_local) * sizeof(int);
    build_grouped_expert_schedule_kernel<<<1, schedule_threads, schedule_smem, stream>>>(
        topk_ids.data_ptr<int64_t>(), topk_slots.data_ptr<int>(),
        expert_counts.data_ptr<int>(), expert_offsets.data_ptr<int>(), a_map.data_ptr<int>(),
        c_map.data_ptr<int>(), row_to_expert.data_ptr<int>(),
        grouped_rows_count.data_ptr<int>(), tile_idx_to_expert_idx.data_ptr<int>(),
        problem_sizes_mnkl.data_ptr<int>(), total_routes, TOPK, E_local,
        static_cast<int>(num_experts), static_cast<int>(local_expert_offset), H, I);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    has_grouped_schedule = true;
  }

  constexpr dim3 block_stage1(16, 4);
  const dim3 grid_stage1((I + block_stage1.x - 1) / block_stage1.x,
                         (total_routes + block_stage1.y - 1) / block_stage1.y);
  if (has_grouped_schedule) {
    const int gemm1_scale_vec = H / gemm1_scale_cols;
    const size_t stage1_smem =
        static_cast<size_t>(3 * block_stage1.y) * sizeof(int) +
        static_cast<size_t>(block_stage1.y) * gemm1_scale_vec * sizeof(__nv_bfloat16);
    stage1_grouped_activation_kernel<<<grid_stage1, block_stage1, stage1_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
        gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
        gemm1_bias.data_ptr<float>(), topk_slots.data_ptr<int>(),
        row_to_expert.data_ptr<int>(), grouped_rows_count.data_ptr<int>(),
        a_map.data_ptr<int>(), mid.data_ptr<float>(), H, I, E_local, gemm1_scale_cols,
        gemm1_bias.numel() > 0);
  } else {
    stage1_activation_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
        gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
        gemm1_bias.data_ptr<float>(), topk_ids.data_ptr<int64_t>(), mid.data_ptr<float>(), T, H,
        I, E_local, TOPK, static_cast<int>(num_experts),
        static_cast<int>(local_expert_offset), gemm1_scale_cols, gemm1_bias.numel() > 0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto out_accum = torch::empty({T, H}, hidden_states.options().dtype(torch::kFloat32));
  C10_CUDA_CHECK(cudaMemsetAsync(out_accum.data_ptr<float>(), 0,
                                 static_cast<size_t>(T) * H * sizeof(float), stream));

  const int split_count = gemm2_scale_cols >= 8 ? 4 : (gemm2_scale_cols >= 2 ? 2 : 1);
  constexpr dim3 block_stage2(16, 4);
  const dim3 grid_stage2((H + block_stage2.x - 1) / block_stage2.x,
                         (T * TOPK + block_stage2.y - 1) / block_stage2.y, split_count);
  stage2_grouped_down_kernel<<<grid_stage2, block_stage2, 0, stream>>>(
      mid.data_ptr<float>(), gemm2_weights.data_ptr<uint8_t>(),
      gemm2_weights_scale.data_ptr<float>(), gemm2_bias.data_ptr<float>(),
      topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
      out_accum.data_ptr<float>(), T, H, I, E_local, TOPK,
      static_cast<int>(num_experts), static_cast<int>(local_expert_offset), gemm2_scale_cols,
      gemm2_bias.numel() > 0, split_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int finalize_block = 256;
  const int total = T * H;
  finalize_output_kernel<<<(total + finalize_block - 1) / finalize_block, finalize_block, 0,
                           stream>>>(
      out_accum.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
