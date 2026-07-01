#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
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

__device__ __forceinline__ uint32_t encode_fp4_e2m1_nearest(float x) {
  const bool neg = x < 0.0f;
  const float ax = fabsf(x);
  uint32_t mag;
  if (ax < 0.25f) {
    mag = 0;
  } else if (ax < 0.75f) {
    mag = 1;
  } else if (ax < 1.25f) {
    mag = 2;
  } else if (ax < 1.75f) {
    mag = 3;
  } else if (ax < 2.5f) {
    mag = 4;
  } else if (ax < 3.5f) {
    mag = 5;
  } else if (ax < 5.0f) {
    mag = 6;
  } else {
    mag = 7;
  }
  return mag | (neg ? 8u : 0u);
}

__device__ __forceinline__ bool is_u32_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0;
}

__device__ __forceinline__ bool is_u128_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

__device__ __forceinline__ int shuffle32_src_to_dst_row(int row) {
  const int in_block = row & 31;
  return (row & ~31) + ((in_block & 3) << 3) + (in_block >> 2);
}

__device__ __forceinline__ int prepared_gemm1_row(int logical_row, int I,
                                                  bool use_prepared_layout) {
  if (!use_prepared_layout) {
    return logical_row;
  }
  const int gated_row =
      (logical_row < I) ? (logical_row << 1) : (((logical_row - I) << 1) + 1);
  return shuffle32_src_to_dst_row(gated_row);
}

__device__ __forceinline__ int prepared_gemm2_row(int logical_row,
                                                  bool use_prepared_layout) {
  return use_prepared_layout ? shuffle32_src_to_dst_row(logical_row) : logical_row;
}

__device__ __forceinline__ int64_t scale_offset_128x4(int row, int col, int cols) {
  const int padded_cols = (cols + 3) & ~3;
  const int column_idx_in_group = col & 3;
  const int column_group_idx = col >> 2;
  const int row_idx_in_group0 = row & 31;
  const int row_idx_in_group1 = (row & 127) >> 5;
  const int row_group_idx = row >> 7;
  return static_cast<int64_t>(row_group_idx) * 128 * padded_cols +
         static_cast<int64_t>(column_group_idx) * 512 +
         static_cast<int64_t>(row_idx_in_group0) * 16 +
         static_cast<int64_t>(row_idx_in_group1) * 4 + column_idx_in_group;
}

__device__ __forceinline__ float load_weight_scale(
    const float* __restrict__ scales,
    int64_t expert_base,
    int row,
    int col,
    int cols,
    bool use_prepared_layout) {
  const int64_t offset =
      use_prepared_layout ? scale_offset_128x4(row, col, cols)
                          : static_cast<int64_t>(row) * cols + col;
  return scales[expert_base + offset];
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

__device__ __forceinline__ void accumulate_fp4_block16_gemm2(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ row,
    int begin,
    float scale,
    float& acc) {
  const uint32_t packed0 = ld_global_u32_l1_no_allocate(row + (begin >> 1));
  const uint32_t packed1 = ld_global_u32_l1_no_allocate(row + (begin >> 1) + 4);
  accumulate_fp4_word8_gemm2(activations, begin, packed0, scale, acc);
  accumulate_fp4_word8_gemm2(activations, begin + 8, packed1, scale, acc);
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

__device__ __forceinline__ float load_fp4_hidden_scaled(
    const uint8_t* __restrict__ hidden,
    const float* __restrict__ hidden_scale,
    int hidden_scale_vec,
    int j) {
  const uint8_t byte = hidden[j >> 1];
  const uint32_t code = (j & 1) ? (byte >> 4) : byte;
  return decode_fp4_e2m1_branchless(code) * hidden_scale[j / hidden_scale_vec];
}

__device__ __forceinline__ void accumulate_fp4_pair_scaled_tile_fp4_hidden(
    const uint8_t* __restrict__ hidden,
    const float* __restrict__ hidden_scale,
    int hidden_scale_vec,
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
    const float hidden_val = load_fp4_hidden_scaled(hidden, hidden_scale, hidden_scale_vec, j);
    acc1 = fmaf(hidden_val, decode_fp4(byte1 >> 4) * scale1, acc1);
    acc2 = fmaf(hidden_val, decode_fp4(byte2 >> 4) * scale2, acc2);
    ++j;
  }

  const uint8_t* ptr1 = row1 + (j >> 1);
  const uint8_t* ptr2 = row2 + (j >> 1);
  while (j + 2 <= end) {
    const uint8_t byte1 = *ptr1;
    const uint8_t byte2 = *ptr2;
    const float hidden0 = load_fp4_hidden_scaled(hidden, hidden_scale, hidden_scale_vec, j);
    const float hidden1 = load_fp4_hidden_scaled(hidden, hidden_scale, hidden_scale_vec, j + 1);
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
    const float hidden_val = load_fp4_hidden_scaled(hidden, hidden_scale, hidden_scale_vec, j);
    acc1 = fmaf(hidden_val, decode_fp4(byte1) * scale1, acc1);
    acc2 = fmaf(hidden_val, decode_fp4(byte2) * scale2, acc2);
  }
}

__device__ __forceinline__ void accumulate_fp4_pair_word8_gemm1_fp4_hidden(
    uint32_t hidden_packed,
    uint32_t packed1,
    uint32_t packed2,
    float scale1,
    float scale2,
    float& acc1,
    float& acc2) {
#pragma unroll
  for (int lane = 0; lane < 8; ++lane) {
    const uint32_t shift = 4u * static_cast<uint32_t>(lane);
    const float hidden_val = decode_fp4_e2m1_branchless(hidden_packed >> shift);
    acc1 = fmaf(hidden_val, decode_fp4_e2m1_branchless(packed1 >> shift) * scale1, acc1);
    acc2 = fmaf(hidden_val, decode_fp4_e2m1_branchless(packed2 >> shift) * scale2, acc2);
  }
}

__device__ __forceinline__ void accumulate_fp4_pair_block16_gemm1_fp4_hidden(
    const uint8_t* __restrict__ hidden,
    const uint8_t* __restrict__ row1,
    const uint8_t* __restrict__ row2,
    int begin,
    float scale1,
    float scale2,
    float& acc1,
    float& acc2) {
  const uint32_t hidden0 = ld_global_u32_l1_no_allocate(hidden + (begin >> 1));
  const uint32_t hidden1 = ld_global_u32_l1_no_allocate(hidden + (begin >> 1) + 4);
  const uint32_t packed10 = ld_global_u32_l1_no_allocate(row1 + (begin >> 1));
  const uint32_t packed11 = ld_global_u32_l1_no_allocate(row1 + (begin >> 1) + 4);
  const uint32_t packed20 = ld_global_u32_l1_no_allocate(row2 + (begin >> 1));
  const uint32_t packed21 = ld_global_u32_l1_no_allocate(row2 + (begin >> 1) + 4);
  accumulate_fp4_pair_word8_gemm1_fp4_hidden(hidden0, packed10, packed20, scale1, scale2,
                                             acc1, acc2);
  accumulate_fp4_pair_word8_gemm1_fp4_hidden(hidden1, packed11, packed21, scale1, scale2,
                                             acc1, acc2);
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

__device__ __forceinline__ void accumulate_fp4_scaled_i_tiles_gemm2_layout(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ row,
    const float* __restrict__ scales,
    int64_t scale_expert_base,
    int scale_row,
    int scale_cols,
    int scale_vec,
    bool use_prepared_layout,
    float& acc) {
  if (!use_prepared_layout) {
    const float* scale_row_ptr = scales + scale_expert_base +
                                 static_cast<int64_t>(scale_row) * scale_cols;
    accumulate_fp4_scaled_i_tiles_gemm2(activations, row, scale_row_ptr, scale_cols,
                                        scale_vec, acc);
    return;
  }

  for (int scale_col = 0; scale_col < scale_cols; ++scale_col) {
    const int begin = scale_col * scale_vec;
    const int end = begin + scale_vec;
    const float scale = load_weight_scale(scales, scale_expert_base, scale_row, scale_col,
                                          scale_cols, true);
    accumulate_fp4_scaled_tile(activations, row, begin, end, scale, acc);
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
    bool has_gemm1_bias,
    bool use_prepared_layout) {
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
  const int w1_x1_row_idx = prepared_gemm1_row(i, I, use_prepared_layout);
  const int w1_x2_row_idx = prepared_gemm1_row(I + i, I, use_prepared_layout);
  const int64_t w1_x1_packed_base =
      ((static_cast<int64_t>(le) * 2 * I + w1_x1_row_idx) * (H / 2));
  const int64_t w1_x2_packed_base =
      ((static_cast<int64_t>(le) * 2 * I + w1_x2_row_idx) * (H / 2));
  const int64_t w1_scale_expert_base =
      static_cast<int64_t>(le) * 2 * I * gemm1_scale_cols;
  const __nv_bfloat16* hidden_row = hidden_states + static_cast<int64_t>(t) * H;
  const uint8_t* w1_x1_row = gemm1_weights + w1_x1_packed_base;
  const uint8_t* w1_x2_row = gemm1_weights + w1_x2_packed_base;

  for (int scale_col = 0; scale_col < gemm1_scale_cols; ++scale_col) {
    const int begin = scale_col * gemm1_scale_vec;
    const int end = begin + gemm1_scale_vec;
    const float scale1 =
        load_weight_scale(gemm1_weights_scale, w1_scale_expert_base, w1_x1_row_idx,
                          scale_col, gemm1_scale_cols, use_prepared_layout);
    const float scale2 =
        load_weight_scale(gemm1_weights_scale, w1_scale_expert_base, w1_x2_row_idx,
                          scale_col, gemm1_scale_cols, use_prepared_layout);
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
    bool has_gemm1_bias,
    bool use_prepared_layout) {
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
  const int w1_x1_row_idx = prepared_gemm1_row(i, I, use_prepared_layout);
  const int w1_x2_row_idx = prepared_gemm1_row(I + i, I, use_prepared_layout);
  const int64_t w1_x1_packed_base =
      ((static_cast<int64_t>(le) * 2 * I + w1_x1_row_idx) * (H / 2));
  const int64_t w1_x2_packed_base =
      ((static_cast<int64_t>(le) * 2 * I + w1_x2_row_idx) * (H / 2));
  const int64_t w1_scale_expert_base =
      static_cast<int64_t>(le) * 2 * I * gemm1_scale_cols;
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
        const float scale1 =
            load_weight_scale(gemm1_weights_scale, w1_scale_expert_base, w1_x1_row_idx,
                              scale_col, gemm1_scale_cols, use_prepared_layout);
        const float scale2 =
            load_weight_scale(gemm1_weights_scale, w1_scale_expert_base, w1_x2_row_idx,
                              scale_col, gemm1_scale_cols, use_prepared_layout);
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
    const int64_t scale1_offset =
        w1_scale_expert_base +
        (use_prepared_layout
             ? scale_offset_128x4(w1_x1_row_idx, scale_col, gemm1_scale_cols)
             : static_cast<int64_t>(w1_x1_row_idx) * gemm1_scale_cols + scale_col);
    const int64_t scale2_offset =
        w1_scale_expert_base +
        (use_prepared_layout
             ? scale_offset_128x4(w1_x2_row_idx, scale_col, gemm1_scale_cols)
             : static_cast<int64_t>(w1_x2_row_idx) * gemm1_scale_cols + scale_col);
    const float scale1 =
        __uint_as_float(ld_global_u32_l1_no_allocate(gemm1_weights_scale + scale1_offset));
    const float scale2 =
        __uint_as_float(ld_global_u32_l1_no_allocate(gemm1_weights_scale + scale2_offset));
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

__global__ void stage1_activation_fp4_kernel(
    const uint8_t* __restrict__ hidden_states,
    const float* __restrict__ hidden_states_scale,
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
    int hidden_scale_cols,
    bool has_gemm1_bias,
    bool use_prepared_layout) {
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
  const int hidden_scale_vec = H / hidden_scale_cols;
  float x1 = has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + i] : 0.0f;
  float x2 = has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + I + i] : 0.0f;
  const int w1_x1_row_idx = prepared_gemm1_row(i, I, use_prepared_layout);
  const int w1_x2_row_idx = prepared_gemm1_row(I + i, I, use_prepared_layout);
  const int64_t w1_x1_packed_base =
      ((static_cast<int64_t>(le) * 2 * I + w1_x1_row_idx) * (H / 2));
  const int64_t w1_x2_packed_base =
      ((static_cast<int64_t>(le) * 2 * I + w1_x2_row_idx) * (H / 2));
  const int64_t w1_scale_expert_base =
      static_cast<int64_t>(le) * 2 * I * gemm1_scale_cols;
  const uint8_t* hidden_row = hidden_states + static_cast<int64_t>(t) * (H / 2);
  const float* hidden_scale_row = hidden_states_scale + static_cast<int64_t>(t) * hidden_scale_cols;
  const uint8_t* w1_x1_row = gemm1_weights + w1_x1_packed_base;
  const uint8_t* w1_x2_row = gemm1_weights + w1_x2_packed_base;

  for (int scale_col = 0; scale_col < gemm1_scale_cols; ++scale_col) {
    const int begin = scale_col * gemm1_scale_vec;
    const int end = begin + gemm1_scale_vec;
    const float scale1 =
        load_weight_scale(gemm1_weights_scale, w1_scale_expert_base, w1_x1_row_idx,
                          scale_col, gemm1_scale_cols, use_prepared_layout);
    const float scale2 =
        load_weight_scale(gemm1_weights_scale, w1_scale_expert_base, w1_x2_row_idx,
                          scale_col, gemm1_scale_cols, use_prepared_layout);
    accumulate_fp4_pair_scaled_tile_fp4_hidden(hidden_row, hidden_scale_row, hidden_scale_vec,
                                               w1_x1_row, w1_x2_row, begin, end, scale1,
                                               scale2, x1, x2);
  }
  mid[static_cast<int64_t>(tk) * I + i] = silu(x2) * x1;
}

__global__ void stage1_activation_fp4_warp_k_kernel(
    const uint8_t* __restrict__ hidden_states,
    const float* __restrict__ hidden_states_scale,
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
    int hidden_scale_cols,
    bool has_gemm1_bias,
    bool use_prepared_layout) {
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
  const int hidden_scale_vec = H / hidden_scale_cols;
  const int w1_x1_row_idx = prepared_gemm1_row(i, I, use_prepared_layout);
  const int w1_x2_row_idx = prepared_gemm1_row(I + i, I, use_prepared_layout);
  const int64_t w1_x1_packed_base =
      ((static_cast<int64_t>(le) * 2 * I + w1_x1_row_idx) * (H / 2));
  const int64_t w1_x2_packed_base =
      ((static_cast<int64_t>(le) * 2 * I + w1_x2_row_idx) * (H / 2));
  const int64_t w1_scale_expert_base =
      static_cast<int64_t>(le) * 2 * I * gemm1_scale_cols;
  const uint8_t* hidden_row = hidden_states + static_cast<int64_t>(t) * (H / 2);
  const float* hidden_scale_row = hidden_states_scale + static_cast<int64_t>(t) * hidden_scale_cols;
  const uint8_t* w1_x1_row = gemm1_weights + w1_x1_packed_base;
  const uint8_t* w1_x2_row = gemm1_weights + w1_x2_packed_base;

  if (gemm1_scale_vec != 16 || hidden_scale_vec != 16 || !is_u32_aligned(w1_x1_row) ||
      !is_u32_aligned(w1_x2_row) || !is_u32_aligned(hidden_row)) {
    if (lane == 0) {
      float x1 = has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + i] : 0.0f;
      float x2 =
          has_gemm1_bias ? gemm1_bias[(static_cast<int64_t>(le) * 2 * I) + I + i] : 0.0f;
      for (int scale_col = 0; scale_col < gemm1_scale_cols; ++scale_col) {
        const int begin = scale_col * gemm1_scale_vec;
        const int end = begin + gemm1_scale_vec;
        const float scale1 =
            load_weight_scale(gemm1_weights_scale, w1_scale_expert_base, w1_x1_row_idx,
                              scale_col, gemm1_scale_cols, use_prepared_layout);
        const float scale2 =
            load_weight_scale(gemm1_weights_scale, w1_scale_expert_base, w1_x2_row_idx,
                              scale_col, gemm1_scale_cols, use_prepared_layout);
        accumulate_fp4_pair_scaled_tile_fp4_hidden(hidden_row, hidden_scale_row,
                                                   hidden_scale_vec, w1_x1_row, w1_x2_row,
                                                   begin, end, scale1, scale2, x1, x2);
      }
      mid[static_cast<int64_t>(tk) * I + i] = silu(x2) * x1;
    }
    return;
  }

  float partial_x1 = 0.0f;
  float partial_x2 = 0.0f;
  for (int scale_col = lane; scale_col < gemm1_scale_cols; scale_col += 32) {
    const int begin = scale_col * 16;
    const float hidden_scale =
        __uint_as_float(ld_global_u32_l1_no_allocate(hidden_scale_row + scale_col));
    const int64_t scale1_offset =
        w1_scale_expert_base +
        (use_prepared_layout
             ? scale_offset_128x4(w1_x1_row_idx, scale_col, gemm1_scale_cols)
             : static_cast<int64_t>(w1_x1_row_idx) * gemm1_scale_cols + scale_col);
    const int64_t scale2_offset =
        w1_scale_expert_base +
        (use_prepared_layout
             ? scale_offset_128x4(w1_x2_row_idx, scale_col, gemm1_scale_cols)
             : static_cast<int64_t>(w1_x2_row_idx) * gemm1_scale_cols + scale_col);
    const float scale1 =
        __uint_as_float(ld_global_u32_l1_no_allocate(gemm1_weights_scale + scale1_offset));
    const float scale2 =
        __uint_as_float(ld_global_u32_l1_no_allocate(gemm1_weights_scale + scale2_offset));
    accumulate_fp4_pair_block16_gemm1_fp4_hidden(
        hidden_row, w1_x1_row, w1_x2_row, begin, scale1 * hidden_scale,
        scale2 * hidden_scale, partial_x1, partial_x2);
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

__global__ void quantize_dequant_mid_fp4_kernel(float* __restrict__ mid, int rows, int I) {
  const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int blocks_per_row = I >> 4;
  const int total_blocks = rows * blocks_per_row;
  if (block_idx >= total_blocks) {
    return;
  }

  const int row = block_idx / blocks_per_row;
  const int col_block = block_idx - row * blocks_per_row;
  float* base = mid + static_cast<int64_t>(row) * I + col_block * 16;

  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    amax = fmaxf(amax, fabsf(base[j]));
  }

  if (amax == 0.0f) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      base[j] = 0.0f;
    }
    return;
  }

  __nv_fp8_e4m3 sf_fp8(amax * (1.0f / 6.0f));
  const float scale = static_cast<float>(sf_fp8);
  if (scale == 0.0f) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      base[j] = 0.0f;
    }
    return;
  }

  const float inv_scale = 1.0f / scale;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const uint32_t code = encode_fp4_e2m1_nearest(base[j] * inv_scale);
    base[j] = decode_fp4_e2m1_branchless(code) * scale;
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
    bool has_gemm2_bias,
    bool use_prepared_layout) {
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
  const int w2_row_idx = prepared_gemm2_row(h, use_prepared_layout);
  const int64_t w2_packed_base = (static_cast<int64_t>(le) * H + w2_row_idx) * (I / 2);
  const int64_t w2_scale_expert_base = static_cast<int64_t>(le) * H * gemm2_scale_cols;
  const uint8_t* w2_row = gemm2_weights + w2_packed_base;

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
    accumulate_fp4_scaled_i_tiles_gemm2_layout(
        mid_row, w2_row, gemm2_weights_scale, w2_scale_expert_base, w2_row_idx,
        gemm2_scale_cols, gemm2_scale_vec, use_prepared_layout, partial);

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

__global__ void stage2_direct_topk_finalize_kernel(
    const float* __restrict__ mid,
    const uint8_t* __restrict__ gemm2_weights,
    const float* __restrict__ gemm2_weights_scale,
    const float* __restrict__ gemm2_bias,
    const int64_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ out,
    int T,
    int H,
    int I,
    int E_local,
    int TOPK,
    int num_experts,
    int local_expert_offset,
    int gemm2_scale_cols,
    bool has_gemm2_bias,
    bool use_prepared_layout) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y * blockDim.y + threadIdx.y;
  if (h >= H || t >= T) {
    return;
  }

  const int gemm2_scale_vec = I / gemm2_scale_cols;
  const int w2_row_idx = prepared_gemm2_row(h, use_prepared_layout);
  float total = 0.0f;

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

    float partial = has_gemm2_bias ? gemm2_bias[static_cast<int64_t>(le) * H + h] : 0.0f;
    const int64_t w2_packed_base =
        (static_cast<int64_t>(le) * H + w2_row_idx) * (I / 2);
    const int64_t w2_scale_expert_base = static_cast<int64_t>(le) * H * gemm2_scale_cols;
    const uint8_t* w2_row = gemm2_weights + w2_packed_base;
    const float* mid_row = mid + static_cast<int64_t>(tk) * I;

    accumulate_fp4_scaled_i_tiles_gemm2_layout(
        mid_row, w2_row, gemm2_weights_scale, w2_scale_expert_base, w2_row_idx,
        gemm2_scale_cols, gemm2_scale_vec, use_prepared_layout, partial);
    total = fmaf(topk_weights[tk], partial, total);
  }

  out[static_cast<int64_t>(t) * H + h] = __float2bfloat16(total);
}

__global__ void stage2_direct_topk_finalize_warp_kernel(
    const float* __restrict__ mid,
    const uint8_t* __restrict__ gemm2_weights,
    const float* __restrict__ gemm2_weights_scale,
    const float* __restrict__ gemm2_bias,
    const int64_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    __nv_bfloat16* __restrict__ out,
    int T,
    int H,
    int I,
    int E_local,
    int TOPK,
    int num_experts,
    int local_expert_offset,
    int gemm2_scale_cols,
    bool has_gemm2_bias,
    bool use_prepared_layout) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int64_t out_linear =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const int64_t total_outputs = static_cast<int64_t>(T) * H;
  if (out_linear >= total_outputs) {
    return;
  }

  const int h = static_cast<int>(out_linear % H);
  const int t = static_cast<int>(out_linear / H);
  const int w2_row_idx = prepared_gemm2_row(h, use_prepared_layout);
  constexpr unsigned int full_warp_mask = 0xFFFFFFFFu;
  float total = 0.0f;

  for (int k = 0; k < TOPK; ++k) {
    const int tk = t * TOPK + k;
    const int64_t global_expert_raw = topk_ids[tk];
    float partial = 0.0f;
    if (global_expert_raw >= 0 && global_expert_raw < num_experts) {
      const int le = static_cast<int>(global_expert_raw) - local_expert_offset;
      if (le >= 0 && le < E_local) {
        const int64_t w2_packed_base =
            (static_cast<int64_t>(le) * H + w2_row_idx) * (I / 2);
        const int64_t w2_scale_expert_base =
            static_cast<int64_t>(le) * H * gemm2_scale_cols;
        const uint8_t* w2_row = gemm2_weights + w2_packed_base;
        const float* mid_row = mid + static_cast<int64_t>(tk) * I;
        for (int scale_col = lane; scale_col < gemm2_scale_cols; scale_col += 32) {
          const float scale = load_weight_scale(
              gemm2_weights_scale, w2_scale_expert_base, w2_row_idx, scale_col,
              gemm2_scale_cols, use_prepared_layout);
          accumulate_fp4_block16_gemm2(mid_row, w2_row, scale_col * 16, scale, partial);
        }
        if (lane == 0 && has_gemm2_bias) {
          partial += gemm2_bias[static_cast<int64_t>(le) * H + h];
        }
      }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(full_warp_mask, partial, offset);
    }
    if (lane == 0) {
      total = fmaf(topk_weights[tk], partial, total);
    }
  }

  if (lane == 0) {
    out[static_cast<int64_t>(t) * H + h] = __float2bfloat16(total);
  }
}

__global__ void routing_topk_softmax_type1_bf16_kernel(
    const __nv_bfloat16* __restrict__ routing_logits,
    int64_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    int T,
    int E,
    int TOPK,
    float scale) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) {
    return;
  }

  float best_vals[16];
  int best_idx[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    best_vals[i] = -3.402823466e+38F;
    best_idx[i] = -1;
  }

  const __nv_bfloat16* row = routing_logits + static_cast<int64_t>(t) * E;
  for (int e = 0; e < E; ++e) {
    const float val = __bfloat162float(row[e]);
    if (val <= best_vals[TOPK - 1]) {
      continue;
    }
    int pos = TOPK - 1;
    while (pos > 0 && val > best_vals[pos - 1]) {
      best_vals[pos] = best_vals[pos - 1];
      best_idx[pos] = best_idx[pos - 1];
      --pos;
    }
    best_vals[pos] = val;
    best_idx[pos] = e;
  }

  const float max_val = best_vals[0];
  float denom = 0.0f;
  for (int i = 0; i < TOPK; ++i) {
    const float w = __expf(best_vals[i] - max_val);
    best_vals[i] = w;
    denom += w;
  }
  const float inv_denom = scale / denom;
  for (int i = 0; i < TOPK; ++i) {
    topk_ids[static_cast<int64_t>(t) * TOPK + i] = static_cast<int64_t>(best_idx[i]);
    topk_weights[static_cast<int64_t>(t) * TOPK + i] = best_vals[i] * inv_denom;
  }
}

__global__ void routing_topk_softmax_type1_f32_kernel(
    const float* __restrict__ routing_logits,
    int64_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    int T,
    int E,
    int TOPK,
    float scale) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) {
    return;
  }

  float best_vals[16];
  int best_idx[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    best_vals[i] = -3.402823466e+38F;
    best_idx[i] = -1;
  }

  const float* row = routing_logits + static_cast<int64_t>(t) * E;
  for (int e = 0; e < E; ++e) {
    const float val = row[e];
    if (val <= best_vals[TOPK - 1]) {
      continue;
    }
    int pos = TOPK - 1;
    while (pos > 0 && val > best_vals[pos - 1]) {
      best_vals[pos] = best_vals[pos - 1];
      best_idx[pos] = best_idx[pos - 1];
      --pos;
    }
    best_vals[pos] = val;
    best_idx[pos] = e;
  }

  const float max_val = best_vals[0];
  float denom = 0.0f;
  for (int i = 0; i < TOPK; ++i) {
    const float w = __expf(best_vals[i] - max_val);
    best_vals[i] = w;
    denom += w;
  }
  const float inv_denom = scale / denom;
  for (int i = 0; i < TOPK; ++i) {
    topk_ids[static_cast<int64_t>(t) * TOPK + i] = static_cast<int64_t>(best_idx[i]);
    topk_weights[static_cast<int64_t>(t) * TOPK + i] = best_vals[i] * inv_denom;
  }
}

__device__ __forceinline__ int32_t pack_bf16_score_idx(float score, int expert_idx) {
  const uint32_t score_bits =
      static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(score)));
  const uint32_t idx_bits = static_cast<uint32_t>(expert_idx) & 0xFFFFu;
  return static_cast<int32_t>((idx_bits << 16) | score_bits);
}

__global__ void routing_topk_softmax_type1_pack_bf16_kernel(
    const __nv_bfloat16* __restrict__ routing_logits,
    int32_t* __restrict__ topk_packed,
    int T,
    int E,
    int TOPK,
    float scale) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) {
    return;
  }

  float best_vals[16];
  int best_idx[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    best_vals[i] = -3.402823466e+38F;
    best_idx[i] = -1;
  }

  const __nv_bfloat16* row = routing_logits + static_cast<int64_t>(t) * E;
  for (int e = 0; e < E; ++e) {
    const float val = __bfloat162float(row[e]);
    if (val <= best_vals[TOPK - 1]) {
      continue;
    }
    int pos = TOPK - 1;
    while (pos > 0 && val > best_vals[pos - 1]) {
      best_vals[pos] = best_vals[pos - 1];
      best_idx[pos] = best_idx[pos - 1];
      --pos;
    }
    best_vals[pos] = val;
    best_idx[pos] = e;
  }

  const float max_val = best_vals[0];
  float denom = 0.0f;
  for (int i = 0; i < TOPK; ++i) {
    const float w = __expf(best_vals[i] - max_val);
    best_vals[i] = w;
    denom += w;
  }
  const float inv_denom = scale / denom;
  for (int i = 0; i < TOPK; ++i) {
    topk_packed[static_cast<int64_t>(t) * TOPK + i] =
        pack_bf16_score_idx(best_vals[i] * inv_denom, best_idx[i]);
  }
}

__global__ void routing_topk_softmax_type1_pack_f32_kernel(
    const float* __restrict__ routing_logits,
    int32_t* __restrict__ topk_packed,
    int T,
    int E,
    int TOPK,
    float scale) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) {
    return;
  }

  float best_vals[16];
  int best_idx[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    best_vals[i] = -3.402823466e+38F;
    best_idx[i] = -1;
  }

  const float* row = routing_logits + static_cast<int64_t>(t) * E;
  for (int e = 0; e < E; ++e) {
    const float val = row[e];
    if (val <= best_vals[TOPK - 1]) {
      continue;
    }
    int pos = TOPK - 1;
    while (pos > 0 && val > best_vals[pos - 1]) {
      best_vals[pos] = best_vals[pos - 1];
      best_idx[pos] = best_idx[pos - 1];
      --pos;
    }
    best_vals[pos] = val;
    best_idx[pos] = e;
  }

  const float max_val = best_vals[0];
  float denom = 0.0f;
  for (int i = 0; i < TOPK; ++i) {
    const float w = __expf(best_vals[i] - max_val);
    best_vals[i] = w;
    denom += w;
  }
  const float inv_denom = scale / denom;
  for (int i = 0; i < TOPK; ++i) {
    topk_packed[static_cast<int64_t>(t) * TOPK + i] =
        pack_bf16_score_idx(best_vals[i] * inv_denom, best_idx[i]);
  }
}

__device__ __forceinline__ int unpack_packed_expert_idx(int32_t packed) {
  const uint32_t raw = static_cast<uint32_t>(packed);
  return static_cast<int>(static_cast<int16_t>((raw >> 16) & 0xFFFFu));
}

__global__ void routing_metadata_chunk_ranks_counts_kernel(
    const int32_t* __restrict__ topk_packed,
    int32_t* __restrict__ expanded_idx_to_permuted_idx,
    int32_t* __restrict__ chunk_counts,
    int total_slots,
    int num_experts,
    int local_expert_offset,
    int local_num_experts) {
  extern __shared__ int smem[];
  int* counts = smem;
  int* chunk_experts = smem + local_num_experts;
  const int tid = threadIdx.x;
  const int chunk_idx = blockIdx.x;
  const int slot = chunk_idx * blockDim.x + tid;

  for (int local_expert = tid; local_expert < local_num_experts; local_expert += blockDim.x) {
    counts[local_expert] = 0;
  }
  __syncthreads();

  int local_expert = -1;
  if (slot < total_slots) {
    const int expert = unpack_packed_expert_idx(topk_packed[slot]);
    local_expert = expert - local_expert_offset;
    if (expert < 0 || expert >= num_experts || local_expert < 0 ||
        local_expert >= local_num_experts) {
      local_expert = -1;
    }
  }
  chunk_experts[tid] = local_expert;
  __syncthreads();

  if (slot < total_slots) {
    if (local_expert >= 0) {
      int rank_in_chunk = 0;
      for (int prev = 0; prev < tid; ++prev) {
        rank_in_chunk += (chunk_experts[prev] == local_expert);
      }
      expanded_idx_to_permuted_idx[slot] = rank_in_chunk;
      atomicAdd(counts + local_expert, 1);
    } else {
      expanded_idx_to_permuted_idx[slot] = -1;
    }
  }
  __syncthreads();

  int32_t* chunk_counts_row =
      chunk_counts + static_cast<int64_t>(chunk_idx) * local_num_experts;
  for (int local_idx = tid; local_idx < local_num_experts; local_idx += blockDim.x) {
    chunk_counts_row[local_idx] = counts[local_idx];
  }
}

__global__ void routing_metadata_chunk_prefix_kernel(
    const int32_t* __restrict__ chunk_counts,
    int32_t* __restrict__ chunk_offsets,
    int32_t* __restrict__ expert_counts,
    int num_chunks,
    int local_num_experts) {
  const int local_expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_expert >= local_num_experts) {
    return;
  }

  int running = 0;
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    const int64_t offset = static_cast<int64_t>(chunk) * local_num_experts + local_expert;
    const int count = chunk_counts[offset];
    chunk_offsets[offset] = running;
    running += count;
  }
  expert_counts[local_expert] = running;
}

__global__ void routing_metadata_prefix_kernel(
    const int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_padded_offsets,
    int32_t* __restrict__ cta_idx_xy_to_batch_idx,
    int32_t* __restrict__ cta_idx_xy_to_mn_limit,
    int32_t* __restrict__ num_non_exiting_ctas,
    int32_t* __restrict__ total_num_padded_tokens,
    int local_num_experts,
    int tile_tokens_dim) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  int padded_running = 0;
  int cta_running = 0;
  for (int local_expert = 0; local_expert < local_num_experts; ++local_expert) {
    const int count = expert_counts[local_expert];
    const int ctas = (count + tile_tokens_dim - 1) / tile_tokens_dim;
    expert_padded_offsets[local_expert] = padded_running;

    for (int cta = 0; cta < ctas; ++cta) {
      const int cta_idx = cta_running + cta;
      cta_idx_xy_to_batch_idx[cta_idx] = local_expert;
      const int padded_cta_limit = (cta_idx + 1) * tile_tokens_dim;
      const int expert_real_limit = padded_running + count;
      cta_idx_xy_to_mn_limit[cta_idx] =
          padded_cta_limit < expert_real_limit ? padded_cta_limit : expert_real_limit;
    }

    padded_running += ctas * tile_tokens_dim;
    cta_running += ctas;
  }
  expert_padded_offsets[local_num_experts] = padded_running;
  num_non_exiting_ctas[0] = cta_running;
  total_num_padded_tokens[0] = padded_running;
}

__global__ void routing_metadata_scatter_kernel(
    const int32_t* __restrict__ topk_packed,
    int32_t* __restrict__ expanded_idx_to_permuted_idx,
    int32_t* __restrict__ permuted_idx_to_token_idx,
    const int32_t* __restrict__ expert_padded_offsets,
    const int32_t* __restrict__ chunk_offsets,
    int total_slots,
    int TOPK,
    int rank_chunk_size,
    int num_experts,
    int local_expert_offset,
    int local_num_experts) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= total_slots) {
    return;
  }

  const int rank_in_chunk = expanded_idx_to_permuted_idx[slot];
  if (rank_in_chunk < 0) {
    return;
  }

  const int expert = unpack_packed_expert_idx(topk_packed[slot]);
  const int local_expert = expert - local_expert_offset;
  if (expert < 0 || expert >= num_experts || local_expert < 0 ||
      local_expert >= local_num_experts) {
    expanded_idx_to_permuted_idx[slot] = -1;
    return;
  }

  const int chunk_idx = slot / rank_chunk_size;
  const int global_rank =
      chunk_offsets[static_cast<int64_t>(chunk_idx) * local_num_experts + local_expert] +
      rank_in_chunk;
  const int permuted_idx = expert_padded_offsets[local_expert] + global_rank;
  expanded_idx_to_permuted_idx[slot] = permuted_idx;
  permuted_idx_to_token_idx[permuted_idx] = slot / TOPK;
}

__global__ void pack_hidden_bmm_from_metadata_kernel(
    const int32_t* __restrict__ topk_packed,
    const int32_t* __restrict__ expanded_idx_to_permuted_idx,
    const int32_t* __restrict__ expert_padded_offsets,
    const uint4* __restrict__ hidden_q,
    const uint4* __restrict__ hidden_scale,
    uint4* __restrict__ hidden_q_bmm,
    uint4* __restrict__ hidden_scale_bmm,
    int total_slots,
    int TOPK,
    int q_chunks_per_row,
    int scale_chunks_per_row,
    int num_experts,
    int local_expert_offset,
    int local_num_experts,
    int padded_rows) {
  const int64_t total = static_cast<int64_t>(total_slots) * q_chunks_per_row;
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }

  const int chunk = static_cast<int>(linear % q_chunks_per_row);
  const int slot = static_cast<int>(linear / q_chunks_per_row);
  const int permuted_idx = expanded_idx_to_permuted_idx[slot];
  if (permuted_idx < 0) {
    return;
  }

  const int expert = unpack_packed_expert_idx(topk_packed[slot]);
  const int local_expert = expert - local_expert_offset;
  if (expert < 0 || expert >= num_experts || local_expert < 0 ||
      local_expert >= local_num_experts) {
    return;
  }

  const int rank = permuted_idx - expert_padded_offsets[local_expert];
  if (rank < 0 || rank >= padded_rows) {
    return;
  }

  const int token = slot / TOPK;
  const int64_t dst_row = static_cast<int64_t>(local_expert) * padded_rows + rank;
  hidden_q_bmm[dst_row * q_chunks_per_row + chunk] =
      hidden_q[static_cast<int64_t>(token) * q_chunks_per_row + chunk];
  if (chunk < scale_chunks_per_row) {
    hidden_scale_bmm[dst_row * scale_chunks_per_row + chunk] =
        hidden_scale[static_cast<int64_t>(token) * scale_chunks_per_row + chunk];
  }
}

__device__ __forceinline__ int nvfp4_swizzled_scale_offset(int row, int scale_col,
                                                           int groups_k) {
  return (row / 128) * groups_k * 512 + (scale_col / 4) * 512 +
         (row % 32) * 16 + ((row % 128) / 32) * 4 + (scale_col % 4);
}

__global__ void nvfp4_block_scale_interleave_kernel(
    const uint8_t* __restrict__ scale,
    uint8_t* __restrict__ swizzled,
    int batches,
    int rows,
    int scale_cols,
    int groups_k,
    int swizzled_bytes_per_batch) {
  const int64_t total = static_cast<int64_t>(batches) * rows * scale_cols;
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }

  const int scale_col = static_cast<int>(linear % scale_cols);
  const int row = static_cast<int>((linear / scale_cols) % rows);
  const int batch = static_cast<int>(linear / (static_cast<int64_t>(rows) * scale_cols));
  const int dst = batch * swizzled_bytes_per_batch +
                  nvfp4_swizzled_scale_offset(row, scale_col, groups_k);
  swizzled[dst] = scale[linear];
}

__global__ void pack_hidden_bmm_swizzled_from_metadata_kernel(
    const int32_t* __restrict__ topk_packed,
    const int32_t* __restrict__ expanded_idx_to_permuted_idx,
    const int32_t* __restrict__ expert_padded_offsets,
    const uint4* __restrict__ hidden_q,
    const uint4* __restrict__ hidden_scale,
    uint4* __restrict__ hidden_q_bmm,
    uint32_t* __restrict__ hidden_scale_swizzled,
    int total_slots,
    int TOPK,
    int q_chunks_per_row,
    int scale_chunks_per_row,
    int scale_cols,
    int groups_k,
    int swizzled_bytes_per_expert,
    int num_experts,
    int local_expert_offset,
    int local_num_experts,
    int padded_rows) {
  const int64_t total = static_cast<int64_t>(total_slots) * q_chunks_per_row;
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }

  const int chunk = static_cast<int>(linear % q_chunks_per_row);
  const int slot = static_cast<int>(linear / q_chunks_per_row);
  const int permuted_idx = expanded_idx_to_permuted_idx[slot];
  if (permuted_idx < 0) {
    return;
  }

  const int expert = unpack_packed_expert_idx(topk_packed[slot]);
  const int local_expert = expert - local_expert_offset;
  if (expert < 0 || expert >= num_experts || local_expert < 0 ||
      local_expert >= local_num_experts) {
    return;
  }

  const int rank = permuted_idx - expert_padded_offsets[local_expert];
  if (rank < 0 || rank >= padded_rows) {
    return;
  }

  const int token = slot / TOPK;
  const int64_t dst_row = static_cast<int64_t>(local_expert) * padded_rows + rank;
  hidden_q_bmm[dst_row * q_chunks_per_row + chunk] =
      hidden_q[static_cast<int64_t>(token) * q_chunks_per_row + chunk];

  if (chunk < scale_chunks_per_row) {
    const uint4 scale_vec =
        hidden_scale[static_cast<int64_t>(token) * scale_chunks_per_row + chunk];
    const int scale_col = chunk * 16;
    const int expert_base_u32 = (local_expert * swizzled_bytes_per_expert) / 4;
    const int dst0 =
        expert_base_u32 + nvfp4_swizzled_scale_offset(rank, scale_col, groups_k) / 4;
    const int dst1 =
        expert_base_u32 + nvfp4_swizzled_scale_offset(rank, scale_col + 4, groups_k) / 4;
    const int dst2 =
        expert_base_u32 + nvfp4_swizzled_scale_offset(rank, scale_col + 8, groups_k) / 4;
    const int dst3 =
        expert_base_u32 + nvfp4_swizzled_scale_offset(rank, scale_col + 12, groups_k) / 4;
    if (scale_col + 12 < scale_cols) {
      hidden_scale_swizzled[dst0] = scale_vec.x;
      hidden_scale_swizzled[dst1] = scale_vec.y;
      hidden_scale_swizzled[dst2] = scale_vec.z;
      hidden_scale_swizzled[dst3] = scale_vec.w;
    }
  }
}

}  // namespace

void routing_topk_softmax_type1_cuda(torch::Tensor routing_logits, torch::Tensor topk_ids,
                                     torch::Tensor topk_weights, int64_t top_k,
                                     double routed_scaling_factor) {
  const int T = static_cast<int>(routing_logits.size(0));
  const int E = static_cast<int>(routing_logits.size(1));
  const int TOPK = static_cast<int>(top_k);
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int block = 128;
  const int grid = (T + block - 1) / block;
  const float scale = static_cast<float>(routed_scaling_factor);
  if (routing_logits.scalar_type() == torch::kBFloat16) {
    routing_topk_softmax_type1_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(routing_logits.data_ptr<at::BFloat16>()),
        topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(), T, E, TOPK, scale);
  } else {
    routing_topk_softmax_type1_f32_kernel<<<grid, block, 0, stream>>>(
        routing_logits.data_ptr<float>(), topk_ids.data_ptr<int64_t>(),
        topk_weights.data_ptr<float>(), T, E, TOPK, scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void routing_topk_softmax_type1_pack_cuda(torch::Tensor routing_logits,
                                          torch::Tensor topk_packed, int64_t top_k,
                                          double routed_scaling_factor) {
  const int T = static_cast<int>(routing_logits.size(0));
  const int E = static_cast<int>(routing_logits.size(1));
  const int TOPK = static_cast<int>(top_k);
  const auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int block = 128;
  const int grid = (T + block - 1) / block;
  const float scale = static_cast<float>(routed_scaling_factor);
  if (routing_logits.scalar_type() == torch::kBFloat16) {
    routing_topk_softmax_type1_pack_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(routing_logits.data_ptr<at::BFloat16>()),
        topk_packed.data_ptr<int32_t>(), T, E, TOPK, scale);
  } else {
    routing_topk_softmax_type1_pack_f32_kernel<<<grid, block, 0, stream>>>(
        routing_logits.data_ptr<float>(), topk_packed.data_ptr<int32_t>(), T, E, TOPK,
        scale);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void routing_metadata_from_packed_cuda(
    torch::Tensor topk_packed, torch::Tensor expanded_idx_to_permuted_idx,
    torch::Tensor permuted_idx_to_token_idx, torch::Tensor cta_idx_xy_to_batch_idx,
    torch::Tensor cta_idx_xy_to_mn_limit, torch::Tensor num_non_exiting_ctas,
    torch::Tensor total_num_padded_tokens, torch::Tensor expert_counts,
    torch::Tensor expert_padded_offsets, torch::Tensor chunk_counts,
    torch::Tensor chunk_offsets, int64_t num_experts, int64_t local_expert_offset,
    int64_t local_num_experts, int64_t tile_tokens_dim) {
  const int T = static_cast<int>(topk_packed.size(0));
  const int TOPK = static_cast<int>(topk_packed.size(1));
  const int total_slots = T * TOPK;
  const auto stream = at::cuda::getCurrentCUDAStream();

  C10_CUDA_CHECK(cudaMemsetAsync(expert_counts.data_ptr<int32_t>(), 0,
                                 expert_counts.numel() * sizeof(int32_t),
                                 stream.stream()));
  C10_CUDA_CHECK(cudaMemsetAsync(expanded_idx_to_permuted_idx.data_ptr<int32_t>(), 0xFF,
                                 expanded_idx_to_permuted_idx.numel() * sizeof(int32_t),
                                 stream.stream()));
  C10_CUDA_CHECK(cudaMemsetAsync(permuted_idx_to_token_idx.data_ptr<int32_t>(), 0xFF,
                                 permuted_idx_to_token_idx.numel() * sizeof(int32_t),
                                 stream.stream()));
  C10_CUDA_CHECK(cudaMemsetAsync(cta_idx_xy_to_batch_idx.data_ptr<int32_t>(), 0xFF,
                                 cta_idx_xy_to_batch_idx.numel() * sizeof(int32_t),
                                 stream.stream()));
  C10_CUDA_CHECK(cudaMemsetAsync(cta_idx_xy_to_mn_limit.data_ptr<int32_t>(), 0xFF,
                                 cta_idx_xy_to_mn_limit.numel() * sizeof(int32_t),
                                 stream.stream()));

  constexpr int block = 256;
  const int rank_shared_bytes =
      (static_cast<int>(local_num_experts) + block) * static_cast<int>(sizeof(int32_t));
  const int num_rank_chunks = static_cast<int>(chunk_counts.size(0));
  if (num_rank_chunks > 0) {
    routing_metadata_chunk_ranks_counts_kernel<<<num_rank_chunks, block, rank_shared_bytes,
                                                  stream>>>(
        topk_packed.data_ptr<int32_t>(), expanded_idx_to_permuted_idx.data_ptr<int32_t>(),
        chunk_counts.data_ptr<int32_t>(), total_slots, static_cast<int>(num_experts),
        static_cast<int>(local_expert_offset), static_cast<int>(local_num_experts));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  constexpr int prefix_block = 128;
  const int prefix_grid =
      (static_cast<int>(local_num_experts) + prefix_block - 1) / prefix_block;
  routing_metadata_chunk_prefix_kernel<<<prefix_grid, prefix_block, 0, stream>>>(
      chunk_counts.data_ptr<int32_t>(), chunk_offsets.data_ptr<int32_t>(),
      expert_counts.data_ptr<int32_t>(), num_rank_chunks, static_cast<int>(local_num_experts));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  routing_metadata_prefix_kernel<<<1, 1, 0, stream>>>(
      expert_counts.data_ptr<int32_t>(), expert_padded_offsets.data_ptr<int32_t>(),
      cta_idx_xy_to_batch_idx.data_ptr<int32_t>(), cta_idx_xy_to_mn_limit.data_ptr<int32_t>(),
      num_non_exiting_ctas.data_ptr<int32_t>(), total_num_padded_tokens.data_ptr<int32_t>(),
      static_cast<int>(local_num_experts), static_cast<int>(tile_tokens_dim));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (total_slots > 0) {
    const int grid = (total_slots + block - 1) / block;
    routing_metadata_scatter_kernel<<<grid, block, 0, stream>>>(
        topk_packed.data_ptr<int32_t>(), expanded_idx_to_permuted_idx.data_ptr<int32_t>(),
        permuted_idx_to_token_idx.data_ptr<int32_t>(), expert_padded_offsets.data_ptr<int32_t>(),
        chunk_offsets.data_ptr<int32_t>(), total_slots, TOPK, block, static_cast<int>(num_experts),
        static_cast<int>(local_expert_offset), static_cast<int>(local_num_experts));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void pack_hidden_bmm_from_metadata_cuda(
    torch::Tensor topk_packed, torch::Tensor expanded_idx_to_permuted_idx,
    torch::Tensor expert_padded_offsets, torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale, torch::Tensor hidden_packed_bmm,
    torch::Tensor hidden_scale_bmm, int64_t num_experts, int64_t local_expert_offset,
    int64_t local_num_experts, int64_t padded_rows) {
  const int T = static_cast<int>(topk_packed.size(0));
  const int TOPK = static_cast<int>(topk_packed.size(1));
  const int total_slots = T * TOPK;
  const int q_chunks_per_row = static_cast<int>(hidden_states.size(1) / 16);
  const int scale_chunks_per_row =
      static_cast<int>((hidden_states_scale.size(1) * hidden_states_scale.element_size()) / 16);
  const int64_t total_chunks = static_cast<int64_t>(total_slots) * q_chunks_per_row;
  if (total_chunks == 0) {
    return;
  }

  constexpr int block = 256;
  const int grid = static_cast<int>((total_chunks + block - 1) / block);
  const auto stream = at::cuda::getCurrentCUDAStream();
  pack_hidden_bmm_from_metadata_kernel<<<grid, block, 0, stream>>>(
      topk_packed.data_ptr<int32_t>(), expanded_idx_to_permuted_idx.data_ptr<int32_t>(),
      expert_padded_offsets.data_ptr<int32_t>(),
      reinterpret_cast<const uint4*>(hidden_states.data_ptr<uint8_t>()),
      reinterpret_cast<const uint4*>(hidden_states_scale.data_ptr()),
      reinterpret_cast<uint4*>(hidden_packed_bmm.data_ptr<uint8_t>()),
      reinterpret_cast<uint4*>(hidden_scale_bmm.data_ptr()), total_slots, TOPK, q_chunks_per_row,
      scale_chunks_per_row, static_cast<int>(num_experts), static_cast<int>(local_expert_offset),
      static_cast<int>(local_num_experts), static_cast<int>(padded_rows));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_block_scale_interleave_cuda(torch::Tensor scale, torch::Tensor swizzled) {
  const int batches = static_cast<int>(scale.size(0));
  const int rows = static_cast<int>(scale.size(1));
  const int scale_cols = static_cast<int>(scale.size(2));
  const int groups_k = (scale_cols + 3) / 4;
  const int swizzled_bytes_per_batch = static_cast<int>(swizzled.size(1));
  const int64_t total = static_cast<int64_t>(batches) * rows * scale_cols;
  if (total == 0) {
    return;
  }
  constexpr int block = 256;
  const int grid = static_cast<int>((total + block - 1) / block);
  const auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(swizzled.data_ptr(), 0, swizzled.numel(),
                                 stream.stream()));
  nvfp4_block_scale_interleave_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(scale.data_ptr()),
      reinterpret_cast<uint8_t*>(swizzled.data_ptr()), batches, rows, scale_cols, groups_k,
      swizzled_bytes_per_batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_hidden_bmm_swizzled_from_metadata_cuda(
    torch::Tensor topk_packed, torch::Tensor expanded_idx_to_permuted_idx,
    torch::Tensor expert_padded_offsets, torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale, torch::Tensor hidden_packed_bmm,
    torch::Tensor hidden_scale_swizzled, int64_t num_experts,
    int64_t local_expert_offset, int64_t local_num_experts, int64_t padded_rows) {
  const int T = static_cast<int>(topk_packed.size(0));
  const int TOPK = static_cast<int>(topk_packed.size(1));
  const int total_slots = T * TOPK;
  const int q_chunks_per_row = static_cast<int>(hidden_states.size(1) / 16);
  const int scale_cols = static_cast<int>(hidden_states_scale.size(1));
  const int scale_chunks_per_row = static_cast<int>(scale_cols / 16);
  const int groups_k = (scale_cols + 3) / 4;
  const int swizzled_bytes_per_expert = static_cast<int>(hidden_scale_swizzled.size(1));
  const int64_t total_chunks = static_cast<int64_t>(total_slots) * q_chunks_per_row;
  if (total_chunks == 0) {
    return;
  }

  constexpr int block = 256;
  const int grid = static_cast<int>((total_chunks + block - 1) / block);
  const auto stream = at::cuda::getCurrentCUDAStream();
  pack_hidden_bmm_swizzled_from_metadata_kernel<<<grid, block, 0, stream>>>(
      topk_packed.data_ptr<int32_t>(), expanded_idx_to_permuted_idx.data_ptr<int32_t>(),
      expert_padded_offsets.data_ptr<int32_t>(),
      reinterpret_cast<const uint4*>(hidden_states.data_ptr<uint8_t>()),
      reinterpret_cast<const uint4*>(hidden_states_scale.data_ptr()),
      reinterpret_cast<uint4*>(hidden_packed_bmm.data_ptr<uint8_t>()),
      reinterpret_cast<uint32_t*>(hidden_scale_swizzled.data_ptr()), total_slots, TOPK,
      q_chunks_per_row, scale_chunks_per_row, scale_cols, groups_k,
      swizzled_bytes_per_expert, static_cast<int>(num_experts),
      static_cast<int>(local_expert_offset), static_cast<int>(local_num_experts),
      static_cast<int>(padded_rows));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_moe_forward_cuda(torch::Tensor hidden_states, torch::Tensor hidden_states_scale,
                            torch::Tensor gemm1_weights, torch::Tensor gemm1_weights_scale,
                            torch::Tensor gemm1_bias, torch::Tensor gemm2_weights,
                            torch::Tensor gemm2_weights_scale, torch::Tensor gemm2_bias,
                            torch::Tensor topk_ids, torch::Tensor topk_weights,
                            torch::Tensor out, int64_t num_experts,
                            int64_t local_expert_offset, int64_t intermediate_size,
                            bool use_prepared_weight_layout) {
  const int T = static_cast<int>(hidden_states.size(0));
  const bool hidden_is_fp4 = hidden_states.scalar_type() == torch::kUInt8;
  const int H = static_cast<int>(hidden_is_fp4 ? hidden_states.size(1) * 2
                                               : hidden_states.size(1));
  const int I = static_cast<int>(intermediate_size);
  const int E_local = static_cast<int>(gemm1_weights.size(0));
  const int TOPK = static_cast<int>(topk_ids.size(1));
  const int gemm1_scale_cols = static_cast<int>(gemm1_weights_scale.size(2));
  const int gemm2_scale_cols = static_cast<int>(gemm2_weights_scale.size(2));
  const int hidden_scale_cols =
      hidden_is_fp4 ? static_cast<int>(hidden_states_scale.size(1)) : 0;

  auto mid = torch::empty({T * TOPK, I}, hidden_states.options().dtype(torch::kFloat32));
  const auto stream = at::cuda::getCurrentCUDAStream();

  const int gemm1_scale_vec = H / gemm1_scale_cols;
  if (hidden_is_fp4) {
    const int hidden_scale_vec = H / hidden_scale_cols;
    if (gemm1_scale_vec == 16 && hidden_scale_vec == 16 && (H % 16) == 0) {
      constexpr int stage1_warp_threads = 256;
      constexpr int stage1_warps_per_block = stage1_warp_threads / 32;
      const int64_t total_stage1_outputs = static_cast<int64_t>(T) * TOPK * I;
      const dim3 block_stage1(stage1_warp_threads);
      const dim3 grid_stage1(
          static_cast<unsigned int>((total_stage1_outputs + stage1_warps_per_block - 1) /
                                    stage1_warps_per_block));
      stage1_activation_fp4_warp_k_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
          hidden_states.data_ptr<uint8_t>(), hidden_states_scale.data_ptr<float>(),
          gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
          gemm1_bias.data_ptr<float>(), topk_ids.data_ptr<int64_t>(), mid.data_ptr<float>(),
          T, H, I, E_local, TOPK, static_cast<int>(num_experts),
          static_cast<int>(local_expert_offset), gemm1_scale_cols, hidden_scale_cols,
          gemm1_bias.numel() > 0, use_prepared_weight_layout);
    } else {
      constexpr dim3 block_stage1(16, 4);
      const dim3 grid_stage1((I + block_stage1.x - 1) / block_stage1.x,
                             (T * TOPK + block_stage1.y - 1) / block_stage1.y);
      stage1_activation_fp4_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
          hidden_states.data_ptr<uint8_t>(), hidden_states_scale.data_ptr<float>(),
          gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
          gemm1_bias.data_ptr<float>(), topk_ids.data_ptr<int64_t>(), mid.data_ptr<float>(),
          T, H, I, E_local, TOPK, static_cast<int>(num_experts),
          static_cast<int>(local_expert_offset), gemm1_scale_cols, hidden_scale_cols,
          gemm1_bias.numel() > 0, use_prepared_weight_layout);
    }
  } else if (gemm1_scale_vec == 32 && (H % 32) == 0) {
    constexpr int stage1_warp_threads = 256;
    constexpr int stage1_warps_per_block = stage1_warp_threads / 32;
    const int64_t total_stage1_outputs = static_cast<int64_t>(T) * TOPK * I;
    const dim3 block_stage1(stage1_warp_threads);
    const dim3 grid_stage1(
        static_cast<unsigned int>((total_stage1_outputs + stage1_warps_per_block - 1) /
                                  stage1_warps_per_block));
    stage1_activation_warp_k_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
        gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
        gemm1_bias.data_ptr<float>(), topk_ids.data_ptr<int64_t>(), mid.data_ptr<float>(), T, H,
        I, E_local, TOPK, static_cast<int>(num_experts),
        static_cast<int>(local_expert_offset), gemm1_scale_cols, gemm1_bias.numel() > 0,
        use_prepared_weight_layout);
  } else {
    constexpr dim3 block_stage1(16, 4);
    const dim3 grid_stage1((I + block_stage1.x - 1) / block_stage1.x,
                           (T * TOPK + block_stage1.y - 1) / block_stage1.y);
    stage1_activation_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
        gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
        gemm1_bias.data_ptr<float>(), topk_ids.data_ptr<int64_t>(), mid.data_ptr<float>(), T, H,
        I, E_local, TOPK, static_cast<int>(num_experts),
        static_cast<int>(local_expert_offset), gemm1_scale_cols, gemm1_bias.numel() > 0,
        use_prepared_weight_layout);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  if (hidden_is_fp4 && use_prepared_weight_layout && (I % 16) == 0) {
    constexpr int qdq_block = 256;
    const int qdq_items = T * TOPK * (I / 16);
    quantize_dequant_mid_fp4_kernel<<<(qdq_items + qdq_block - 1) / qdq_block, qdq_block, 0,
                                      stream>>>(mid.data_ptr<float>(), T * TOPK, I);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  const int gemm2_scale_vec = I / gemm2_scale_cols;
  if (gemm2_scale_vec == 16 && (I % 16) == 0) {
    constexpr int stage2_warp_threads = 256;
    constexpr int stage2_warps_per_block = stage2_warp_threads / 32;
    const int64_t total_stage2_outputs = static_cast<int64_t>(T) * H;
    const dim3 block_stage2(stage2_warp_threads);
    const dim3 grid_stage2(
        static_cast<unsigned int>((total_stage2_outputs + stage2_warps_per_block - 1) /
                                  stage2_warps_per_block));
    stage2_direct_topk_finalize_warp_kernel<<<grid_stage2, block_stage2, 0, stream>>>(
        mid.data_ptr<float>(), gemm2_weights.data_ptr<uint8_t>(),
        gemm2_weights_scale.data_ptr<float>(), gemm2_bias.data_ptr<float>(),
        topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), T, H, I, E_local,
        TOPK, static_cast<int>(num_experts), static_cast<int>(local_expert_offset),
        gemm2_scale_cols, gemm2_bias.numel() > 0, use_prepared_weight_layout);
  } else {
    constexpr dim3 block_stage2(32, 4);
    const dim3 grid_stage2((H + block_stage2.x - 1) / block_stage2.x,
                           (T + block_stage2.y - 1) / block_stage2.y);
    stage2_direct_topk_finalize_kernel<<<grid_stage2, block_stage2, 0, stream>>>(
        mid.data_ptr<float>(), gemm2_weights.data_ptr<uint8_t>(),
        gemm2_weights_scale.data_ptr<float>(), gemm2_bias.data_ptr<float>(),
        topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), T, H, I, E_local,
        TOPK, static_cast<int>(num_experts), static_cast<int>(local_expert_offset),
        gemm2_scale_cols, gemm2_bias.numel() > 0, use_prepared_weight_layout);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
