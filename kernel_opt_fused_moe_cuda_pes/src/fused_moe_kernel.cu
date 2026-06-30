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

__global__ void fp4_moe_forward_kernel(
    const __nv_bfloat16* __restrict__ hidden_states,
    const uint8_t* __restrict__ gemm1_weights,
    const float* __restrict__ gemm1_weights_scale,
    const float* __restrict__ gemm1_bias,
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
    int gemm1_scale_cols,
    int gemm2_scale_cols,
    bool has_gemm1_bias,
    bool has_gemm2_bias) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= T || h >= H) {
    return;
  }

  const int gemm1_scale_vec = H / gemm1_scale_cols;
  const int gemm2_scale_vec = I / gemm2_scale_cols;
  float acc_out = 0.0f;

  for (int k = 0; k < TOPK; ++k) {
    const int64_t global_expert_raw = topk_ids[t * TOPK + k];
    if (global_expert_raw < 0 || global_expert_raw >= num_experts) {
      continue;
    }
    const int le = static_cast<int>(global_expert_raw) - local_expert_offset;
    if (le < 0 || le >= E_local) {
      continue;
    }

    float expert_sum = 0.0f;
    for (int i = 0; i < I; ++i) {
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

      const float activated = silu(x2) * x1;
      const int64_t w2_packed_base = (static_cast<int64_t>(le) * H + h) * (I / 2);
      const int64_t w2_scale_base = (static_cast<int64_t>(le) * H + h) * gemm2_scale_cols;
      const float w2 = load_fp4_scaled(gemm2_weights, gemm2_weights_scale, w2_packed_base,
                                       w2_scale_base, i, gemm2_scale_vec);
      expert_sum = fmaf(activated, w2, expert_sum);
    }

    if (has_gemm2_bias) {
      expert_sum += gemm2_bias[static_cast<int64_t>(le) * H + h];
    }
    acc_out = fmaf(topk_weights[t * TOPK + k], expert_sum, acc_out);
  }

  out[static_cast<int64_t>(t) * H + h] = __float2bfloat16(acc_out);
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

  constexpr dim3 block(16, 4);
  const dim3 grid((H + block.x - 1) / block.x, (T + block.y - 1) / block.y);
  const auto stream = at::cuda::getCurrentCUDAStream();
  fp4_moe_forward_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
      gemm1_weights.data_ptr<uint8_t>(), gemm1_weights_scale.data_ptr<float>(),
      gemm1_bias.data_ptr<float>(), gemm2_weights.data_ptr<uint8_t>(),
      gemm2_weights_scale.data_ptr<float>(), gemm2_bias.data_ptr<float>(),
      topk_ids.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), T, H, I, E_local, TOPK,
      static_cast<int>(num_experts), static_cast<int>(local_expert_offset), gemm1_scale_cols,
      gemm2_scale_cols, gemm1_bias.numel() > 0, gemm2_bias.numel() > 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
