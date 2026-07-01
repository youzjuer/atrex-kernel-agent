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

__global__ void stage2_split_partial_kernel(
    const float* __restrict__ mid,
    const uint8_t* __restrict__ gemm2_weights,
    const float* __restrict__ gemm2_weights_scale,
    const int64_t* __restrict__ topk_ids,
    float* __restrict__ split_partials,
    int T,
    int H,
    int I,
    int E_local,
    int TOPK,
    int num_experts,
    int local_expert_offset,
    int gemm2_scale_cols,
    int split_count) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int tk = blockIdx.y * blockDim.y + threadIdx.y;
  const int split = blockIdx.z;
  if (tk >= T * TOPK || h >= H || split >= split_count) {
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
  split_partials[(static_cast<int64_t>(tk) * split_count + split) * H + h] = partial;
}

__global__ void stage2_reduce_finalize_kernel(
    const float* __restrict__ split_partials,
    const float* __restrict__ gemm2_bias,
    const int64_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ out,
    int T,
    int H,
    int E_local,
    int TOPK,
    int num_experts,
    int local_expert_offset,
    bool has_gemm2_bias,
    int split_count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = T * H;
  if (idx < total) {
    const int t = idx / H;
    const int h = idx - t * H;
    float accum = 0.0f;
    for (int k = 0; k < TOPK; ++k) {
      const int tk = t * TOPK + k;
      const int64_t global_expert_raw = topk_ids[tk];
      if (global_expert_raw < 0 || global_expert_raw >= num_experts) {
        continue;
      }
      const int le = static_cast<int>(global_expert_raw) - local_expert_offset;
      if (le < 0 || le >= E_local) {
        continue;
      }

      float partial = 0.0f;
      for (int split = 0; split < split_count; ++split) {
        partial += split_partials[(static_cast<int64_t>(tk) * split_count + split) * H + h];
      }
      if (has_gemm2_bias) {
        partial += gemm2_bias[static_cast<int64_t>(le) * H + h];
      }
      accum = fmaf(topk_weights[tk], partial, accum);
    }
    out[idx] = __float2bfloat16(accum);
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

  auto mid = torch::empty({T * TOPK, I}, hidden_states.options().dtype(torch::kFloat32));
  const auto stream = at::cuda::getCurrentCUDAStream();

  constexpr dim3 block_stage1(16, 4);
  const dim3 grid_stage1((I + block_stage1.x - 1) / block_stage1.x,
                         (T * TOPK + block_stage1.y - 1) / block_stage1.y);
  stage1_activation_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
      gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
      gemm1_bias.data_ptr<float>(), topk_ids.data_ptr<int64_t>(), mid.data_ptr<float>(), T, H, I,
      E_local, TOPK, static_cast<int>(num_experts), static_cast<int>(local_expert_offset),
      gemm1_scale_cols, gemm1_bias.numel() > 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int split_count = gemm2_scale_cols >= 8 ? 4 : (gemm2_scale_cols >= 2 ? 2 : 1);
  auto split_partials =
      torch::empty({T * TOPK, split_count, H}, hidden_states.options().dtype(torch::kFloat32));

  constexpr dim3 block_stage2(16, 4);
  const dim3 grid_stage2((H + block_stage2.x - 1) / block_stage2.x,
                         (T * TOPK + block_stage2.y - 1) / block_stage2.y, split_count);
  stage2_split_partial_kernel<<<grid_stage2, block_stage2, 0, stream>>>(
      mid.data_ptr<float>(), gemm2_weights.data_ptr<uint8_t>(),
      gemm2_weights_scale.data_ptr<float>(), topk_ids.data_ptr<int64_t>(),
      split_partials.data_ptr<float>(), T, H, I, E_local, TOPK,
      static_cast<int>(num_experts), static_cast<int>(local_expert_offset), gemm2_scale_cols,
      split_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int reduce_block = 256;
  const int total = T * H;
  stage2_reduce_finalize_kernel<<<(total + reduce_block - 1) / reduce_block, reduce_block, 0,
                                  stream>>>(
      split_partials.data_ptr<float>(), gemm2_bias.data_ptr<float>(), topk_ids.data_ptr<int64_t>(),
      topk_weights.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      T, H, E_local, TOPK, static_cast<int>(num_experts), static_cast<int>(local_expert_offset),
      gemm2_bias.numel() > 0, split_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
