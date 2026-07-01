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

__device__ __forceinline__ void accumulate_fp4_pair_activation_gemm2(
    float activation,
    uint32_t code0,
    uint32_t code1,
    float scale0,
    float scale1,
    float& acc0,
    float& acc1) {
  acc0 = fmaf(activation, decode_fp4_e2m1_branchless(code0) * scale0, acc0);
  acc1 = fmaf(activation, decode_fp4_e2m1_branchless(code1) * scale1, acc1);
}

__device__ __forceinline__ void accumulate_fp4_pair_word8_gemm2(
    const float* __restrict__ activations,
    int base,
    uint32_t packed0,
    uint32_t packed1,
    float scale0,
    float scale1,
    float& acc0,
    float& acc1) {
  const uint4 act0 = ld_global_v4_u32_l1_evict_last(activations + base);
  const uint4 act1 = ld_global_v4_u32_l1_evict_last(activations + base + 4);
  accumulate_fp4_pair_activation_gemm2(__uint_as_float(act0.x), packed0, packed1, scale0,
                                       scale1, acc0, acc1);
  accumulate_fp4_pair_activation_gemm2(__uint_as_float(act0.y), packed0 >> 4, packed1 >> 4,
                                       scale0, scale1, acc0, acc1);
  accumulate_fp4_pair_activation_gemm2(__uint_as_float(act0.z), packed0 >> 8, packed1 >> 8,
                                       scale0, scale1, acc0, acc1);
  accumulate_fp4_pair_activation_gemm2(__uint_as_float(act0.w), packed0 >> 12, packed1 >> 12,
                                       scale0, scale1, acc0, acc1);
  accumulate_fp4_pair_activation_gemm2(__uint_as_float(act1.x), packed0 >> 16, packed1 >> 16,
                                       scale0, scale1, acc0, acc1);
  accumulate_fp4_pair_activation_gemm2(__uint_as_float(act1.y), packed0 >> 20, packed1 >> 20,
                                       scale0, scale1, acc0, acc1);
  accumulate_fp4_pair_activation_gemm2(__uint_as_float(act1.z), packed0 >> 24, packed1 >> 24,
                                       scale0, scale1, acc0, acc1);
  accumulate_fp4_pair_activation_gemm2(__uint_as_float(act1.w), packed0 >> 28, packed1 >> 28,
                                       scale0, scale1, acc0, acc1);
}

__device__ __forceinline__ void accumulate_fp4_pair_block32_gemm2(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    int begin,
    float scale0,
    float scale1,
    float& acc0,
    float& acc1) {
  const uint4 packed0 = ld_global_v4_u32_l1_no_allocate(row0 + (begin >> 1));
  const uint4 packed1 = ld_global_v4_u32_l1_no_allocate(row1 + (begin >> 1));
  accumulate_fp4_pair_word8_gemm2(activations, begin, packed0.x, packed1.x, scale0,
                                  scale1, acc0, acc1);
  accumulate_fp4_pair_word8_gemm2(activations, begin + 8, packed0.y, packed1.y, scale0,
                                  scale1, acc0, acc1);
  accumulate_fp4_pair_word8_gemm2(activations, begin + 16, packed0.z, packed1.z, scale0,
                                  scale1, acc0, acc1);
  accumulate_fp4_pair_word8_gemm2(activations, begin + 24, packed0.w, packed1.w, scale0,
                                  scale1, acc0, acc1);
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

__device__ __forceinline__ void accumulate_fp4_pair_scaled_tile_gemm2(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    int begin,
    int end,
    float scale0,
    float scale1,
    float& acc0,
    float& acc1) {
  int i = begin;
  if (i < end && (i & 1)) {
    const uint8_t byte0 = row0[i >> 1];
    const uint8_t byte1 = row1[i >> 1];
    const float activation = activations[i];
    acc0 = fmaf(activation, decode_fp4(byte0 >> 4) * scale0, acc0);
    acc1 = fmaf(activation, decode_fp4(byte1 >> 4) * scale1, acc1);
    ++i;
  }

  const uint8_t* ptr0 = row0 + (i >> 1);
  const uint8_t* ptr1 = row1 + (i >> 1);
  if (is_u32_aligned(ptr0) && is_u32_aligned(ptr1)) {
    while (i + 8 <= end) {
      const uint32_t packed0 = __ldg(reinterpret_cast<const uint32_t*>(ptr0));
      const uint32_t packed1 = __ldg(reinterpret_cast<const uint32_t*>(ptr1));
#pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        const float activation = activations[i + lane];
        acc0 = fmaf(activation, decode_fp4(packed0 >> (4 * lane)) * scale0, acc0);
        acc1 = fmaf(activation, decode_fp4(packed1 >> (4 * lane)) * scale1, acc1);
      }
      i += 8;
      ptr0 += 4;
      ptr1 += 4;
    }
  }

  while (i + 2 <= end) {
    const uint8_t byte0 = *ptr0;
    const uint8_t byte1 = *ptr1;
    const float activation0 = activations[i];
    const float activation1 = activations[i + 1];
    acc0 = fmaf(activation0, decode_fp4(byte0) * scale0, acc0);
    acc1 = fmaf(activation0, decode_fp4(byte1) * scale1, acc1);
    acc0 = fmaf(activation1, decode_fp4(byte0 >> 4) * scale0, acc0);
    acc1 = fmaf(activation1, decode_fp4(byte1 >> 4) * scale1, acc1);
    i += 2;
    ++ptr0;
    ++ptr1;
  }

  if (i < end) {
    const uint8_t byte0 = *ptr0;
    const uint8_t byte1 = *ptr1;
    const float activation = activations[i];
    acc0 = fmaf(activation, decode_fp4(byte0) * scale0, acc0);
    acc1 = fmaf(activation, decode_fp4(byte1) * scale1, acc1);
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

__device__ __forceinline__ void accumulate_fp4_pair_scaled_i_tiles_gemm2(
    const float* __restrict__ activations,
    const uint8_t* __restrict__ row0,
    const uint8_t* __restrict__ row1,
    const float* __restrict__ scales0,
    const float* __restrict__ scales1,
    int scale_cols,
    int scale_vec,
    float& acc0,
    float& acc1) {
  if (scale_vec != 32 || !is_u128_aligned(row0) || !is_u128_aligned(row1) ||
      !is_u128_aligned(scales0) || !is_u128_aligned(scales1) ||
      !is_u128_aligned(activations)) {
    for (int scale_col = 0; scale_col < scale_cols; ++scale_col) {
      const int begin = scale_col * scale_vec;
      const int end = begin + scale_vec;
      accumulate_fp4_pair_scaled_tile_gemm2(activations, row0, row1, begin, end,
                                            scales0[scale_col], scales1[scale_col],
                                            acc0, acc1);
    }
    return;
  }

  int scale_col = 0;
  for (; scale_col + 4 <= scale_cols; scale_col += 4) {
    const float4 scale4_0 = ld_global_v4_f32_l1_no_allocate(scales0 + scale_col);
    const float4 scale4_1 = ld_global_v4_f32_l1_no_allocate(scales1 + scale_col);
    const int begin = scale_col * 32;
    accumulate_fp4_pair_block32_gemm2(activations, row0, row1, begin, scale4_0.x,
                                      scale4_1.x, acc0, acc1);
    accumulate_fp4_pair_block32_gemm2(activations, row0, row1, begin + 32, scale4_0.y,
                                      scale4_1.y, acc0, acc1);
    accumulate_fp4_pair_block32_gemm2(activations, row0, row1, begin + 64, scale4_0.z,
                                      scale4_1.z, acc0, acc1);
    accumulate_fp4_pair_block32_gemm2(activations, row0, row1, begin + 96, scale4_0.w,
                                      scale4_1.w, acc0, acc1);
  }

  for (; scale_col < scale_cols; ++scale_col) {
    const float scale0 = __uint_as_float(ld_global_u32_l1_no_allocate(scales0 + scale_col));
    const float scale1 = __uint_as_float(ld_global_u32_l1_no_allocate(scales1 + scale_col));
    accumulate_fp4_pair_block32_gemm2(activations, row0, row1, scale_col * 32,
                                      scale0, scale1, acc0, acc1);
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

__device__ __forceinline__ void finalize_stage2_output(
    float weighted,
    int expected,
    int64_t out_idx,
    float* __restrict__ out_accum,
    int* __restrict__ completion_counts,
    __nv_bfloat16* __restrict__ out) {
  if (expected == 1) {
    out[out_idx] = __float2bfloat16(weighted);
    return;
  }

  atomicAdd(out_accum + out_idx, weighted);
  __threadfence();
  const int done = atomicAdd(completion_counts + out_idx, 1) + 1;
  if (done == expected) {
    out[out_idx] = __float2bfloat16(out_accum[out_idx]);
  }
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
  const int h = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const int h_next = h + 1;
  const int le = blockIdx.y;
  if (h >= H) {
    return;
  }
  const bool has_h_next = h_next < H;
  const int expert_count = expert_counts[le];
  if (expert_count == 0) {
    return;
  }

  const int gemm2_scale_vec = I / gemm2_scale_cols;
  const int64_t w2_packed_base0 = (static_cast<int64_t>(le) * H + h) * (I / 2);
  const int64_t w2_scale_base0 = (static_cast<int64_t>(le) * H + h) * gemm2_scale_cols;
  const int64_t w2_packed_base1 =
      (static_cast<int64_t>(le) * H + (has_h_next ? h_next : h)) * (I / 2);
  const int64_t w2_scale_base1 =
      (static_cast<int64_t>(le) * H + (has_h_next ? h_next : h)) * gemm2_scale_cols;
  const uint8_t* w2_row0 = gemm2_weights + w2_packed_base0;
  const uint8_t* w2_row1 = gemm2_weights + w2_packed_base1;
  const float* w2_scale_row0 = gemm2_weights_scale + w2_scale_base0;
  const float* w2_scale_row1 = gemm2_weights_scale + w2_scale_base1;

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

    float partial0 = has_gemm2_bias ? gemm2_bias[static_cast<int64_t>(le) * H + h] : 0.0f;
    float partial1 = (has_h_next && has_gemm2_bias)
                         ? gemm2_bias[static_cast<int64_t>(le) * H + h_next]
                         : 0.0f;
    const float* mid_row = mid + static_cast<int64_t>(tk) * I;
    accumulate_fp4_pair_scaled_i_tiles_gemm2(mid_row, w2_row0, w2_row1, w2_scale_row0,
                                             w2_scale_row1, gemm2_scale_cols,
                                             gemm2_scale_vec, partial0, partial1);

    const float route_weight = topk_weights[tk];
    const int64_t out_idx0 = static_cast<int64_t>(t) * H + h;
    finalize_stage2_output(route_weight * partial0, expected, out_idx0, out_accum,
                           completion_counts, out);
    if (has_h_next) {
      const int64_t out_idx1 = static_cast<int64_t>(t) * H + h_next;
      finalize_stage2_output(route_weight * partial1, expected, out_idx1, out_accum,
                             completion_counts, out);
    }
  }
}

__global__ void stage2_grouped_down_finalize_warp_i_kernel(
    const float* __restrict__ mid,
    const uint8_t* __restrict__ gemm2_weights,
    const float* __restrict__ gemm2_weights_scale,
    const float* __restrict__ gemm2_bias,
    const float* __restrict__ topk_weights,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ grouped_slots,
    const int* __restrict__ token_counts,
    float* __restrict__ out_accum,
    int* __restrict__ completion_counts,
    __nv_bfloat16* __restrict__ out,
    int T,
    int H,
    int I,
    int TOPK,
    int E_local,
    int gemm2_scale_cols,
    bool has_gemm2_bias) {
  constexpr unsigned int full_warp_mask = 0xFFFFFFFFu;
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int64_t output_idx =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const int64_t total_slot_outputs = static_cast<int64_t>(T) * TOPK * H;
  if (output_idx >= total_slot_outputs) {
    return;
  }

  const int h = static_cast<int>(output_idx % H);
  const int slot_pos = static_cast<int>(output_idx / H);
  const int total_local_slots = expert_offsets[E_local];
  if (slot_pos >= total_local_slots) {
    return;
  }

  int le = -1;
  if (lane == 0) {
    int probe = 0;
    while (probe < E_local && slot_pos >= expert_offsets[probe + 1]) {
      ++probe;
    }
    le = (probe < E_local) ? probe : -1;
  }
  le = __shfl_sync(full_warp_mask, le, 0);
  if (le < 0) {
    return;
  }

  const int tk = grouped_slots[slot_pos];
  if (tk < 0 || tk >= T * TOPK) {
    return;
  }
  const int t = tk / TOPK;
  const int expected = token_counts[t];
  if (expected <= 0) {
    return;
  }

  const int64_t w2_packed_base = (static_cast<int64_t>(le) * H + h) * (I / 2);
  const int64_t w2_scale_base = (static_cast<int64_t>(le) * H + h) * gemm2_scale_cols;
  const float* mid_row = mid + static_cast<int64_t>(tk) * I;
  const uint8_t* w2_row = gemm2_weights + w2_packed_base;
  const float* w2_scale_row = gemm2_weights_scale + w2_scale_base;

  if (!is_u128_aligned(mid_row) || !is_u128_aligned(w2_row) ||
      !is_u128_aligned(w2_scale_row)) {
    if (lane == 0) {
      float partial =
          has_gemm2_bias ? gemm2_bias[static_cast<int64_t>(le) * H + h] : 0.0f;
      accumulate_fp4_scaled_i_tiles_gemm2(mid_row, w2_row, w2_scale_row, gemm2_scale_cols,
                                          I / gemm2_scale_cols, partial);
      const float route_weight = topk_weights[tk];
      const int64_t out_idx = static_cast<int64_t>(t) * H + h;
      finalize_stage2_output(route_weight * partial, expected, out_idx, out_accum,
                             completion_counts, out);
    }
    return;
  }

  float partial = 0.0f;
  const int scale_col = lane;
  const float scale =
      __uint_as_float(ld_global_u32_l1_no_allocate(w2_scale_row + scale_col));
  accumulate_fp4_block32_gemm2(mid_row, w2_row, scale_col * 32, scale, partial);

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(full_warp_mask, partial, offset);
  }

  if (lane == 0) {
    if (has_gemm2_bias) {
      partial += gemm2_bias[static_cast<int64_t>(le) * H + h];
    }
    const float route_weight = topk_weights[tk];
    const int64_t out_idx = static_cast<int64_t>(t) * H + h;
    finalize_stage2_output(route_weight * partial, expected, out_idx, out_accum,
                           completion_counts, out);
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

  const int gemm1_scale_vec = H / gemm1_scale_cols;
  if (gemm1_scale_vec == 32 && (H % 32) == 0) {
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
        static_cast<int>(local_expert_offset), gemm1_scale_cols, gemm1_bias.numel() > 0);
  } else {
    constexpr dim3 block_stage1(16, 4);
    const dim3 grid_stage1((I + block_stage1.x - 1) / block_stage1.x,
                           (T * TOPK + block_stage1.y - 1) / block_stage1.y);
    stage1_activation_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
        gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
        gemm1_bias.data_ptr<float>(), topk_ids.data_ptr<int64_t>(), mid.data_ptr<float>(), T, H,
        I, E_local, TOPK, static_cast<int>(num_experts),
        static_cast<int>(local_expert_offset), gemm1_scale_cols, gemm1_bias.numel() > 0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int total_slots = T * TOPK;
  auto int_options = hidden_states.options().dtype(torch::kInt32);
  auto expert_counts = torch::empty({E_local}, int_options);
  auto expert_offsets = torch::empty({E_local + 1}, int_options);
  auto expert_cursors = torch::empty({E_local}, int_options);
  auto grouped_slots = torch::empty({total_slots}, int_options);
  auto token_counts = torch::empty({T}, int_options);
  auto completion_counts = torch::empty({T, H}, int_options);
  auto out_accum = torch::empty({T, H}, hidden_states.options().dtype(torch::kFloat32));

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

  const int gemm2_scale_vec = I / gemm2_scale_cols;
  if (I == 1024 && gemm2_scale_vec == 32 && gemm2_scale_cols == 32) {
    constexpr int stage2_warp_threads = 256;
    constexpr int stage2_warps_per_block = stage2_warp_threads / 32;
    const int64_t total_stage2_outputs = static_cast<int64_t>(T) * TOPK * H;
    const dim3 block_stage2_warp(stage2_warp_threads);
    const dim3 grid_stage2_warp(
        static_cast<unsigned int>((total_stage2_outputs + stage2_warps_per_block - 1) /
                                  stage2_warps_per_block));
    stage2_grouped_down_finalize_warp_i_kernel<<<grid_stage2_warp, block_stage2_warp, 0,
                                                  stream>>>(
        mid.data_ptr<float>(), gemm2_weights.data_ptr<uint8_t>(),
        gemm2_weights_scale.data_ptr<float>(), gemm2_bias.data_ptr<float>(),
        topk_weights.data_ptr<float>(), expert_offsets.data_ptr<int>(),
        grouped_slots.data_ptr<int>(), token_counts.data_ptr<int>(), out_accum.data_ptr<float>(),
        completion_counts.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), T, H, I, TOPK,
        E_local, gemm2_scale_cols, gemm2_bias.numel() > 0);
  } else {
    constexpr dim3 block_stage2(32);
    const dim3 grid_stage2((H + block_stage2.x * 2 - 1) / (block_stage2.x * 2), E_local);
    stage2_grouped_down_finalize_kernel<<<grid_stage2, block_stage2, 0, stream>>>(
        mid.data_ptr<float>(), gemm2_weights.data_ptr<uint8_t>(),
        gemm2_weights_scale.data_ptr<float>(), gemm2_bias.data_ptr<float>(),
        topk_weights.data_ptr<float>(), expert_offsets.data_ptr<int>(),
        expert_counts.data_ptr<int>(), grouped_slots.data_ptr<int>(),
        token_counts.data_ptr<int>(), out_accum.data_ptr<float>(),
        completion_counts.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), T, H, I, TOPK,
        gemm2_scale_cols, gemm2_bias.numel() > 0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
