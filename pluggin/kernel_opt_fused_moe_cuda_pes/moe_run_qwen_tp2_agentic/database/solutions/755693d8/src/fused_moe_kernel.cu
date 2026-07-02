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

__device__ __forceinline__ bool is_u128_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

__device__ __forceinline__ uint4 ld_global_v4_u32_l1_no_allocate(const void* ptr) {
  uint4 out;
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
               : "l"(ptr));
  return out;
}

__device__ __forceinline__ uint4 ld_global_v4_u32_l1_evict_last(const void* ptr) {
  uint4 out;
  asm volatile("ld.global.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
               : "l"(ptr));
  return out;
}

__device__ __forceinline__ uint32_t ld_global_u32_l1_no_allocate(const void* ptr) {
  uint32_t out;
  asm volatile("ld.global.L1::no_allocate.b32 %0, [%1];" : "=r"(out) : "l"(ptr));
  return out;
}

__device__ __forceinline__ float4 ld_global_v4_f32_l1_no_allocate(const float* ptr) {
  const uint4 packed = ld_global_v4_u32_l1_no_allocate(ptr);
  return make_float4(__uint_as_float(packed.x), __uint_as_float(packed.y),
                     __uint_as_float(packed.z), __uint_as_float(packed.w));
}

__device__ __forceinline__ float decode_fp4_e2m1_branchless(uint32_t code) {
  const uint32_t nibble = code & 0x0Fu;
  const uint32_t sign = (nibble & 0x08u) << 28;
  const uint32_t mag = nibble & 0x07u;
  const uint32_t exp = mag >> 1;
  const uint32_t mant = mag & 0x01u;
  const uint32_t normal_bits = sign | ((126u + exp) << 23) | (mant << 22);
  const uint32_t subnormal_bits = sign | (mant * 0x3F000000u);
  const uint32_t normal_mask = 0u - static_cast<uint32_t>(exp != 0u);
  return __uint_as_float((normal_bits & normal_mask) | (subnormal_bits & ~normal_mask));
}

__device__ __forceinline__ void accumulate_fp4_word8_gemm2(
    const float* __restrict__ activations,
    int base,
    uint32_t packed,
    float scale,
    float& acc) {
  const uint4 act0 = ld_global_v4_u32_l1_evict_last(activations + base);
  const uint4 act1 = ld_global_v4_u32_l1_evict_last(activations + base + 4);
  acc = fmaf(__uint_as_float(act0.x), decode_fp4_e2m1_branchless(packed) * scale, acc);
  acc = fmaf(__uint_as_float(act0.y), decode_fp4_e2m1_branchless(packed >> 4) * scale, acc);
  acc = fmaf(__uint_as_float(act0.z), decode_fp4_e2m1_branchless(packed >> 8) * scale, acc);
  acc = fmaf(__uint_as_float(act0.w), decode_fp4_e2m1_branchless(packed >> 12) * scale, acc);
  acc = fmaf(__uint_as_float(act1.x), decode_fp4_e2m1_branchless(packed >> 16) * scale, acc);
  acc = fmaf(__uint_as_float(act1.y), decode_fp4_e2m1_branchless(packed >> 20) * scale, acc);
  acc = fmaf(__uint_as_float(act1.z), decode_fp4_e2m1_branchless(packed >> 24) * scale, acc);
  acc = fmaf(__uint_as_float(act1.w), decode_fp4_e2m1_branchless(packed >> 28) * scale, acc);
}

__device__ __forceinline__ void accumulate_fp4_block32_gemm2(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ row,
    int begin,
    float scale,
    float& acc) {
  const uint4 packed = ld_global_v4_u32_l1_no_allocate(row + (begin >> 1));
  accumulate_fp4_word8_gemm2(activations, begin, packed.x, scale, acc);
  accumulate_fp4_word8_gemm2(activations, begin + 8, packed.y, scale, acc);
  accumulate_fp4_word8_gemm2(activations, begin + 16, packed.z, scale, acc);
  accumulate_fp4_word8_gemm2(activations, begin + 24, packed.w, scale, acc);
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

__device__ __forceinline__ void accumulate_fp4_pair_bf16_bits_gemm1(
    uint32_t hidden_bits,
    uint32_t code1,
    uint32_t code2,
    float scale1,
    float scale2,
    float& acc1,
    float& acc2) {
  const float hidden_val =
      __bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(hidden_bits)));
  acc1 = fmaf(hidden_val, decode_fp4_e2m1_branchless(code1) * scale1, acc1);
  acc2 = fmaf(hidden_val, decode_fp4_e2m1_branchless(code2) * scale2, acc2);
}

__device__ __forceinline__ void accumulate_fp4_pair_word8_gemm1(
    const __nv_bfloat16* __restrict__ hidden,
    int base,
    uint32_t packed1,
    uint32_t packed2,
    float scale1,
    float scale2,
    float& acc1,
    float& acc2) {
  const uint4 hidden_pack = ld_global_v4_u32_l1_evict_last(hidden + base);
  accumulate_fp4_pair_bf16_bits_gemm1(hidden_pack.x, packed1, packed2, scale1, scale2,
                                      acc1, acc2);
  accumulate_fp4_pair_bf16_bits_gemm1(hidden_pack.x >> 16, packed1 >> 4, packed2 >> 4,
                                      scale1, scale2, acc1, acc2);
  accumulate_fp4_pair_bf16_bits_gemm1(hidden_pack.y, packed1 >> 8, packed2 >> 8, scale1,
                                      scale2, acc1, acc2);
  accumulate_fp4_pair_bf16_bits_gemm1(hidden_pack.y >> 16, packed1 >> 12, packed2 >> 12,
                                      scale1, scale2, acc1, acc2);
  accumulate_fp4_pair_bf16_bits_gemm1(hidden_pack.z, packed1 >> 16, packed2 >> 16, scale1,
                                      scale2, acc1, acc2);
  accumulate_fp4_pair_bf16_bits_gemm1(hidden_pack.z >> 16, packed1 >> 20, packed2 >> 20,
                                      scale1, scale2, acc1, acc2);
  accumulate_fp4_pair_bf16_bits_gemm1(hidden_pack.w, packed1 >> 24, packed2 >> 24, scale1,
                                      scale2, acc1, acc2);
  accumulate_fp4_pair_bf16_bits_gemm1(hidden_pack.w >> 16, packed1 >> 28, packed2 >> 28,
                                      scale1, scale2, acc1, acc2);
}

__device__ __forceinline__ void accumulate_fp4_pair_block32_gemm1(
    const __nv_bfloat16* __restrict__ hidden,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ row2,
    int begin,
    float scale1,
    float scale2,
    float& acc1,
    float& acc2) {
  const uint4 packed1 = ld_global_v4_u32_l1_no_allocate(row1 + (begin >> 1));
  const uint4 packed2 = ld_global_v4_u32_l1_no_allocate(row2 + (begin >> 1));
  accumulate_fp4_pair_word8_gemm1(hidden, begin, packed1.x, packed2.x, scale1, scale2,
                                  acc1, acc2);
  accumulate_fp4_pair_word8_gemm1(hidden, begin + 8, packed1.y, packed2.y, scale1,
                                  scale2, acc1, acc2);
  accumulate_fp4_pair_word8_gemm1(hidden, begin + 16, packed1.z, packed2.z, scale1,
                                  scale2, acc1, acc2);
  accumulate_fp4_pair_word8_gemm1(hidden, begin + 24, packed1.w, packed2.w, scale1,
                                  scale2, acc1, acc2);
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

__device__ __forceinline__ void accumulate_fp4_scaled_i_tiles_gemm2(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ row,
    const float* __restrict__ scales,
    int scale_cols,
    int scale_vec,
    float& acc) {
  if (scale_vec != 32 || !is_u128_aligned(row) || !is_u128_aligned(scales) ||
      !is_u128_aligned(activations)) {
    for (int scale_col = 0; scale_col < scale_cols; ++scale_col) {
      const int begin = scale_col * scale_vec;
      const int end = begin + scale_vec;
      accumulate_fp4_scaled_tile(activations, row, begin, end, scales[scale_col], acc);
    }
    return;
  }

  int scale_col = 0;
  for (; scale_col + 4 <= scale_cols; scale_col += 4) {
    const float4 scale4 = ld_global_v4_f32_l1_no_allocate(scales + scale_col);
    const int begin = scale_col * 32;
    accumulate_fp4_block32_gemm2(activations, row, begin, scale4.x, acc);
    accumulate_fp4_block32_gemm2(activations, row, begin + 32, scale4.y, acc);
    accumulate_fp4_block32_gemm2(activations, row, begin + 64, scale4.z, acc);
    accumulate_fp4_block32_gemm2(activations, row, begin + 96, scale4.w, acc);
  }

  for (; scale_col < scale_cols; ++scale_col) {
    const float scale = __uint_as_float(ld_global_u32_l1_no_allocate(scales + scale_col));
    accumulate_fp4_block32_gemm2(activations, row, scale_col * 32, scale, acc);
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

__global__ void stage1_activation_warp_k_kernel(
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
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int64_t out_idx =
      (static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block);
  const int64_t total_outputs = static_cast<int64_t>(T) * TOPK * I;
  if (out_idx >= total_outputs) {
    return;
  }

  const int i = static_cast<int>(out_idx % I);
  const int tk = static_cast<int>(out_idx / I);
  const int t = tk / TOPK;
  const int64_t global_expert_raw = topk_ids[tk];
  if (global_expert_raw < 0 || global_expert_raw >= num_experts) {
    if (lane == 0) {
      mid[static_cast<int64_t>(tk) * I + i] = 0.0f;
    }
    return;
  }
  const int le = static_cast<int>(global_expert_raw) - local_expert_offset;
  if (le < 0 || le >= E_local) {
    if (lane == 0) {
      mid[static_cast<int64_t>(tk) * I + i] = 0.0f;
    }
    return;
  }

  const int gemm1_scale_vec = H / gemm1_scale_cols;
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

  if (gemm1_scale_vec != 32 || !is_u128_aligned(w1_x1_row) ||
      !is_u128_aligned(w1_x2_row) || !is_u128_aligned(hidden_row)) {
    if (lane == 0) {
      float x1 = has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + i] : 0.0f;
      float x2 =
          has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + I + i] : 0.0f;
      for (int scale_col = 0; scale_col < gemm1_scale_cols; ++scale_col) {
        const int begin = scale_col * gemm1_scale_vec;
        const int end = begin + gemm1_scale_vec;
        const float scale1 = gemm1_weights_scale[w1_x1_scale_base + scale_col];
        const float scale2 = gemm1_weights_scale[w1_x2_scale_base + scale_col];
        accumulate_fp4_pair_scaled_tile(hidden_row, w1_x1_row, w1_x2_row, begin, end,
                                        scale1, scale2, x1, x2);
      }
      mid[static_cast<int64_t>(tk) * I + i] = silu(x2) * x1;
    }
    return;
  }

  float partial_x1 = 0.0f;
  float partial_x2 = 0.0f;
  for (int scale_col = lane; scale_col < gemm1_scale_cols; scale_col += 32) {
    const int begin = scale_col * 32;
    const float scale1 =
        __uint_as_float(ld_global_u32_l1_no_allocate(gemm1_weights_scale +
                                                     w1_x1_scale_base + scale_col));
    const float scale2 =
        __uint_as_float(ld_global_u32_l1_no_allocate(gemm1_weights_scale +
                                                     w1_x2_scale_base + scale_col));
    accumulate_fp4_pair_block32_gemm1(hidden_row, w1_x1_row, w1_x2_row, begin, scale1,
                                      scale2, partial_x1, partial_x2);
  }

  constexpr unsigned int full_warp_mask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    partial_x1 += __shfl_down_sync(full_warp_mask, partial_x1, offset);
    partial_x2 += __shfl_down_sync(full_warp_mask, partial_x2, offset);
  }

  if (lane == 0) {
    const float bias1 =
        has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + i] : 0.0f;
    const float bias2 =
        has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + I + i] : 0.0f;
    const float x1 = partial_x1 + bias1;
    const float x2 = partial_x2 + bias2;
    mid[static_cast<int64_t>(tk) * I + i] = silu(x2) * x1;
  }
}

__global__ void prepare_single_token_topk_local_experts_kernel(
    const int64_t* __restrict__ topk_ids,
    int* __restrict__ slot_to_local_expert,
    int* __restrict__ local_slot_count,
    int TOPK,
    int E_local,
    int num_experts,
    int local_expert_offset) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int count = 0;
  for (int k = 0; k < TOPK; ++k) {
    int le = -1;
    const int64_t global_expert_raw = topk_ids[k];
    if (global_expert_raw >= 0 && global_expert_raw < num_experts) {
      const int candidate = static_cast<int>(global_expert_raw) - local_expert_offset;
      if (candidate >= 0 && candidate < E_local) {
        le = candidate;
        ++count;
      }
    }
    slot_to_local_expert[k] = le;
  }
  local_slot_count[0] = count;
}

__global__ void stage1_activation_fixed_topk_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const uint8_t* __restrict__ gemm1_weights,
    const float* __restrict__ gemm1_weights_scale,
    const float* __restrict__ gemm1_bias,
    const int* __restrict__ slot_to_local_expert,
    float* __restrict__ mid,
    int H,
    int I,
    int E_local,
    int TOPK,
    int gemm1_scale_cols,
    bool has_gemm1_bias) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int tk = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= I || tk >= TOPK) {
    return;
  }
  const int le = slot_to_local_expert[tk];
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
  const uint8_t* w1_x1_row = gemm1_weights + w1_x1_packed_base;
  const uint8_t* w1_x2_row = gemm1_weights + w1_x2_packed_base;

  for (int scale_col = 0; scale_col < gemm1_scale_cols; ++scale_col) {
    const int begin = scale_col * gemm1_scale_vec;
    const int end = begin + gemm1_scale_vec;
    const float scale1 = gemm1_weights_scale[w1_x1_scale_base + scale_col];
    const float scale2 = gemm1_weights_scale[w1_x2_scale_base + scale_col];
    accumulate_fp4_pair_scaled_tile(hidden_states, w1_x1_row, w1_x2_row, begin, end,
                                    scale1, scale2, x1, x2);
  }
  mid[static_cast<int64_t>(tk) * I + i] = silu(x2) * x1;
}

__global__ void stage1_activation_warp_k_fixed_topk_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const uint8_t* __restrict__ gemm1_weights,
    const float* __restrict__ gemm1_weights_scale,
    const float* __restrict__ gemm1_bias,
    const int* __restrict__ slot_to_local_expert,
    float* __restrict__ mid,
    int H,
    int I,
    int E_local,
    int TOPK,
    int gemm1_scale_cols,
    bool has_gemm1_bias) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int64_t out_idx =
      (static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block);
  const int64_t total_outputs = static_cast<int64_t>(TOPK) * I;
  if (out_idx >= total_outputs) {
    return;
  }

  const int i = static_cast<int>(out_idx % I);
  const int tk = static_cast<int>(out_idx / I);
  const int le = slot_to_local_expert[tk];
  if (le < 0 || le >= E_local) {
    if (lane == 0) {
      mid[static_cast<int64_t>(tk) * I + i] = 0.0f;
    }
    return;
  }

  const int gemm1_scale_vec = H / gemm1_scale_cols;
  const int64_t w1_x1_packed_base = ((static_cast<int64_t>(le) * 2 * I + i) * (H / 2));
  const int64_t w1_x2_packed_base =
      ((static_cast<int64_t>(le) * 2 * I + I + i) * (H / 2));
  const int64_t w1_x1_scale_base =
      ((static_cast<int64_t>(le) * 2 * I + i) * gemm1_scale_cols);
  const int64_t w1_x2_scale_base =
      ((static_cast<int64_t>(le) * 2 * I + I + i) * gemm1_scale_cols);
  const uint8_t* w1_x1_row = gemm1_weights + w1_x1_packed_base;
  const uint8_t* w1_x2_row = gemm1_weights + w1_x2_packed_base;

  if (gemm1_scale_vec != 32 || !is_u128_aligned(w1_x1_row) ||
      !is_u128_aligned(w1_x2_row) || !is_u128_aligned(hidden_states)) {
    if (lane == 0) {
      float x1 = has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + i] : 0.0f;
      float x2 =
          has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + I + i] : 0.0f;
      for (int scale_col = 0; scale_col < gemm1_scale_cols; ++scale_col) {
        const int begin = scale_col * gemm1_scale_vec;
        const int end = begin + gemm1_scale_vec;
        const float scale1 = gemm1_weights_scale[w1_x1_scale_base + scale_col];
        const float scale2 = gemm1_weights_scale[w1_x2_scale_base + scale_col];
        accumulate_fp4_pair_scaled_tile(hidden_states, w1_x1_row, w1_x2_row, begin, end,
                                        scale1, scale2, x1, x2);
      }
      mid[static_cast<int64_t>(tk) * I + i] = silu(x2) * x1;
    }
    return;
  }

  float partial_x1 = 0.0f;
  float partial_x2 = 0.0f;
  for (int scale_col = lane; scale_col < gemm1_scale_cols; scale_col += 32) {
    const int begin = scale_col * 32;
    const float scale1 =
        __uint_as_float(ld_global_u32_l1_no_allocate(gemm1_weights_scale +
                                                     w1_x1_scale_base + scale_col));
    const float scale2 =
        __uint_as_float(ld_global_u32_l1_no_allocate(gemm1_weights_scale +
                                                     w1_x2_scale_base + scale_col));
    accumulate_fp4_pair_block32_gemm1(hidden_states, w1_x1_row, w1_x2_row, begin, scale1,
                                      scale2, partial_x1, partial_x2);
  }

  constexpr unsigned int full_warp_mask = 0xFFFFFFFFu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    partial_x1 += __shfl_down_sync(full_warp_mask, partial_x1, offset);
    partial_x2 += __shfl_down_sync(full_warp_mask, partial_x2, offset);
  }

  if (lane == 0) {
    const float bias1 =
        has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + i] : 0.0f;
    const float bias2 =
        has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + I + i] : 0.0f;
    const float x1 = partial_x1 + bias1;
    const float x2 = partial_x2 + bias2;
    mid[static_cast<int64_t>(tk) * I + i] = silu(x2) * x1;
  }
}

__global__ void count_local_slots_kernel(
    const int64_t* __restrict__ topk_ids,
    int* __restrict__ expert_counts,
    int* __restrict__ token_counts,
    int total_slots,
    int TOPK,
    int E_local,
    int num_experts,
    int local_expert_offset) {
  const int tk = blockIdx.x * blockDim.x + threadIdx.x;
  if (tk >= total_slots) {
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
  atomicAdd(expert_counts + le, 1);
  atomicAdd(token_counts + (tk / TOPK), 1);
}

__global__ void build_expert_offsets_kernel(
    const int* __restrict__ expert_counts,
    int* __restrict__ expert_offsets,
    int* __restrict__ expert_cursors,
    int E_local) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int running = 0;
  for (int le = 0; le < E_local; ++le) {
    expert_offsets[le] = running;
    expert_cursors[le] = running;
    running += expert_counts[le];
  }
  expert_offsets[E_local] = running;
}

__global__ void fill_grouped_slots_kernel(
    const int64_t* __restrict__ topk_ids,
    int* __restrict__ expert_cursors,
    int* __restrict__ grouped_slots,
    int total_slots,
    int E_local,
    int num_experts,
    int local_expert_offset) {
  const int tk = blockIdx.x * blockDim.x + threadIdx.x;
  if (tk >= total_slots) {
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
  const int pos = atomicAdd(expert_cursors + le, 1);
  grouped_slots[pos] = tk;
}

__global__ void stage2_grouped_down_finalize_kernel(
    const float* __restrict__ mid,
    const uint8_t* __restrict__ gemm2_weights,
    const float* __restrict__ gemm2_weights_scale,
    const float* __restrict__ gemm2_bias,
    const float* __restrict__ topk_weights,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ expert_counts,
    const int* __restrict__ grouped_slots,
    const int* __restrict__ token_counts,
    float* __restrict__ out_accum,
    int* __restrict__ completion_counts,
    __nv_bfloat16* __restrict__ out,
    int T,
    int H,
    int I,
    int TOPK,
    int gemm2_scale_cols,
    bool has_gemm2_bias) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int le = blockIdx.y;
  if (h >= H) {
    return;
  }
  const int expert_count = expert_counts[le];
  if (expert_count == 0) {
    return;
  }

  const int gemm2_scale_vec = I / gemm2_scale_cols;
  const int64_t w2_packed_base = (static_cast<int64_t>(le) * H + h) * (I / 2);
  const int64_t w2_scale_base = (static_cast<int64_t>(le) * H + h) * gemm2_scale_cols;
  const uint8_t* w2_row = gemm2_weights + w2_packed_base;
  const float* w2_scale_row = gemm2_weights_scale + w2_scale_base;

  const int slot_begin = expert_offsets[le];
  for (int row = 0; row < expert_count; ++row) {
    const int tk = grouped_slots[slot_begin + row];
    if (tk < 0 || tk >= T * TOPK) {
      continue;
    }
    const int t = tk / TOPK;
    const int expected = token_counts[t];
    if (expected <= 0) {
      continue;
    }

    float partial = has_gemm2_bias ? gemm2_bias[static_cast<int64_t>(le) * H + h] : 0.0f;
    const float* mid_row = mid + static_cast<int64_t>(tk) * I;
    accumulate_fp4_scaled_i_tiles_gemm2(mid_row, w2_row, w2_scale_row, gemm2_scale_cols,
                                        gemm2_scale_vec, partial);

    const float weighted = topk_weights[tk] * partial;
    const int64_t out_idx = static_cast<int64_t>(t) * H + h;
    if (expected == 1) {
      out[out_idx] = __float2bfloat16(weighted);
      continue;
    }

    atomicAdd(out_accum + out_idx, weighted);
    __threadfence();
    const int done = atomicAdd(completion_counts + out_idx, 1) + 1;
    if (done == expected) {
      out[out_idx] = __float2bfloat16(out_accum[out_idx]);
    }
  }
}

__global__ void stage2_single_token_slots_down_partials_kernel(
    const float* __restrict__ mid,
    const uint8_t* __restrict__ gemm2_weights,
    const float* __restrict__ gemm2_weights_scale,
    const int* __restrict__ slot_to_local_expert,
    float* __restrict__ slot_partials,
    int H,
    int I,
    int E_local,
    int TOPK,
    int gemm2_scale_cols) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int tk = blockIdx.y;
  if (h >= H) {
    return;
  }
  if (tk >= TOPK) {
    return;
  }
  const int le = slot_to_local_expert[tk];
  if (le < 0 || le >= E_local) {
    return;
  }

  const int gemm2_scale_vec = I / gemm2_scale_cols;
  const int64_t w2_packed_base = (static_cast<int64_t>(le) * H + h) * (I / 2);
  const int64_t w2_scale_base = (static_cast<int64_t>(le) * H + h) * gemm2_scale_cols;
  const uint8_t* w2_row = gemm2_weights + w2_packed_base;
  const float* w2_scale_row = gemm2_weights_scale + w2_scale_base;

  float partial = 0.0f;
  const float* mid_row = mid + static_cast<int64_t>(tk) * I;
  accumulate_fp4_scaled_i_tiles_gemm2(mid_row, w2_row, w2_scale_row, gemm2_scale_cols,
                                      gemm2_scale_vec, partial);
  slot_partials[static_cast<int64_t>(tk) * H + h] = partial;
}

__global__ void finalize_single_token_slots_kernel(
    const float* __restrict__ slot_partials,
    const float* __restrict__ gemm2_bias,
    const float* __restrict__ topk_weights,
    const int* __restrict__ slot_to_local_expert,
    __nv_bfloat16* __restrict__ out,
    int H,
    int E_local,
    int TOPK,
    bool has_gemm2_bias) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) {
    return;
  }

  float acc = 0.0f;
  for (int tk = 0; tk < TOPK; ++tk) {
    const int le = slot_to_local_expert[tk];
    if (le < 0 || le >= E_local) {
      continue;
    }
    float partial = slot_partials[static_cast<int64_t>(tk) * H + h];
    if (has_gemm2_bias) {
      partial += gemm2_bias[static_cast<int64_t>(le) * H + h];
    }
    acc = fmaf(topk_weights[tk], partial, acc);
  }
  out[h] = __float2bfloat16(acc);
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
  const int total_slots = T * TOPK;
  auto int_options = hidden_states.options().dtype(torch::kInt32);
  const bool use_fixed_topk_metadata = (T == 1 && TOPK <= 10);
  torch::Tensor slot_to_local_expert;
  torch::Tensor local_slot_count;

  if (use_fixed_topk_metadata) {
    slot_to_local_expert = torch::empty({TOPK}, int_options);
    local_slot_count = torch::empty({1}, int_options);
    prepare_single_token_topk_local_experts_kernel<<<1, 1, 0, stream>>>(
        topk_ids.data_ptr<int64_t>(), slot_to_local_expert.data_ptr<int>(),
        local_slot_count.data_ptr<int>(), TOPK, E_local, static_cast<int>(num_experts),
        static_cast<int>(local_expert_offset));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int gemm1_scale_vec = H / gemm1_scale_cols;
  if (gemm1_scale_vec == 32 && (H % 32) == 0) {
    constexpr int stage1_warp_threads = 256;
    constexpr int stage1_warps_per_block = stage1_warp_threads / 32;
    const int64_t total_stage1_outputs =
        use_fixed_topk_metadata ? static_cast<int64_t>(TOPK) * I
                                : static_cast<int64_t>(T) * TOPK * I;
    const dim3 block_stage1(stage1_warp_threads);
    const dim3 grid_stage1(
        static_cast<unsigned int>((total_stage1_outputs + stage1_warps_per_block - 1) /
                                  stage1_warps_per_block));
    if (use_fixed_topk_metadata) {
      stage1_activation_warp_k_fixed_topk_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
          gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
          gemm1_bias.data_ptr<float>(), slot_to_local_expert.data_ptr<int>(),
          mid.data_ptr<float>(), H, I, E_local, TOPK, gemm1_scale_cols,
          gemm1_bias.numel() > 0);
    } else {
      stage1_activation_warp_k_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
          gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
          gemm1_bias.data_ptr<float>(), topk_ids.data_ptr<int64_t>(), mid.data_ptr<float>(),
          T, H, I, E_local, TOPK, static_cast<int>(num_experts),
          static_cast<int>(local_expert_offset), gemm1_scale_cols, gemm1_bias.numel() > 0);
    }
  } else {
    constexpr dim3 block_stage1(16, 4);
    const int stage1_slots = use_fixed_topk_metadata ? TOPK : T * TOPK;
    const dim3 grid_stage1((I + block_stage1.x - 1) / block_stage1.x,
                           (stage1_slots + block_stage1.y - 1) / block_stage1.y);
    if (use_fixed_topk_metadata) {
      stage1_activation_fixed_topk_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
          gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
          gemm1_bias.data_ptr<float>(), slot_to_local_expert.data_ptr<int>(),
          mid.data_ptr<float>(), H, I, E_local, TOPK, gemm1_scale_cols,
          gemm1_bias.numel() > 0);
    } else {
      stage1_activation_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
          gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
          gemm1_bias.data_ptr<float>(), topk_ids.data_ptr<int64_t>(), mid.data_ptr<float>(),
          T, H, I, E_local, TOPK, static_cast<int>(num_experts),
          static_cast<int>(local_expert_offset), gemm1_scale_cols, gemm1_bias.numel() > 0);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (use_fixed_topk_metadata) {
    auto slot_partials =
        torch::empty({TOPK, H}, hidden_states.options().dtype(torch::kFloat32));

    constexpr dim3 block_stage2(32);
    const dim3 grid_stage2((H + block_stage2.x - 1) / block_stage2.x, TOPK);
    stage2_single_token_slots_down_partials_kernel<<<grid_stage2, block_stage2, 0,
                                                     stream>>>(
        mid.data_ptr<float>(), gemm2_weights.data_ptr<uint8_t>(),
        gemm2_weights_scale.data_ptr<float>(), slot_to_local_expert.data_ptr<int>(),
        slot_partials.data_ptr<float>(), H, I, E_local, TOPK, gemm2_scale_cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    constexpr int finalize_block = 256;
    finalize_single_token_slots_kernel<<<(H + finalize_block - 1) / finalize_block,
                                         finalize_block, 0, stream>>>(
        slot_partials.data_ptr<float>(), gemm2_bias.data_ptr<float>(),
        topk_weights.data_ptr<float>(), slot_to_local_expert.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), H, E_local, TOPK,
        gemm2_bias.numel() > 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  auto completion_counts = torch::empty({T, H}, int_options);
  auto out_accum = torch::empty({T, H}, hidden_states.options().dtype(torch::kFloat32));
  auto expert_counts = torch::empty({E_local}, int_options);
  auto expert_offsets = torch::empty({E_local + 1}, int_options);
  auto expert_cursors = torch::empty({E_local}, int_options);
  auto grouped_slots = torch::empty({total_slots}, int_options);
  auto token_counts = torch::empty({T}, int_options);

  C10_CUDA_CHECK(cudaMemsetAsync(expert_counts.data_ptr<int>(), 0,
                                 static_cast<size_t>(E_local) * sizeof(int), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(token_counts.data_ptr<int>(), 0,
                                 static_cast<size_t>(T) * sizeof(int), stream));
  constexpr int metadata_block = 256;
  count_local_slots_kernel<<<(total_slots + metadata_block - 1) / metadata_block,
                             metadata_block, 0, stream>>>(
      topk_ids.data_ptr<int64_t>(), expert_counts.data_ptr<int>(),
      token_counts.data_ptr<int>(), total_slots, TOPK, E_local,
      static_cast<int>(num_experts), static_cast<int>(local_expert_offset));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  build_expert_offsets_kernel<<<1, 1, 0, stream>>>(
      expert_counts.data_ptr<int>(), expert_offsets.data_ptr<int>(),
      expert_cursors.data_ptr<int>(), E_local);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fill_grouped_slots_kernel<<<(total_slots + metadata_block - 1) / metadata_block,
                              metadata_block, 0, stream>>>(
      topk_ids.data_ptr<int64_t>(), expert_cursors.data_ptr<int>(),
      grouped_slots.data_ptr<int>(), total_slots, E_local, static_cast<int>(num_experts),
      static_cast<int>(local_expert_offset));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  C10_CUDA_CHECK(cudaMemsetAsync(out_accum.data_ptr<float>(), 0,
                                 static_cast<size_t>(T) * H * sizeof(float), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(completion_counts.data_ptr<int>(), 0,
                                 static_cast<size_t>(T) * H * sizeof(int), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(out.data_ptr<at::BFloat16>(), 0,
                                 static_cast<size_t>(T) * H * sizeof(at::BFloat16), stream));

  constexpr dim3 block_stage2(32);
  const dim3 grid_stage2((H + block_stage2.x - 1) / block_stage2.x, E_local);
  stage2_grouped_down_finalize_kernel<<<grid_stage2, block_stage2, 0, stream>>>(
      mid.data_ptr<float>(), gemm2_weights.data_ptr<uint8_t>(),
      gemm2_weights_scale.data_ptr<float>(), gemm2_bias.data_ptr<float>(),
      topk_weights.data_ptr<float>(), expert_offsets.data_ptr<int>(),
      expert_counts.data_ptr<int>(), grouped_slots.data_ptr<int>(), token_counts.data_ptr<int>(),
      out_accum.data_ptr<float>(), completion_counts.data_ptr<int>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), T, H, I, TOPK,
      gemm2_scale_cols, gemm2_bias.numel() > 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
