#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

__global__ void fused_moe_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    const int64_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    float* __restrict__ out,
    int M,
    int H,
    int I,
    int E,
    int TOPK) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || h >= H) {
    return;
  }

  float acc_out = 0.0f;
  for (int k = 0; k < TOPK; ++k) {
    const int64_t expert_id_raw = topk_ids[m * TOPK + k];
    if (expert_id_raw < 0 || expert_id_raw >= E) {
      continue;
    }
    const int e = static_cast<int>(expert_id_raw);
    const float route_w = topk_weights[m * TOPK + k];

    float expert_sum = 0.0f;
    for (int i = 0; i < I; ++i) {
      float gate = 0.0f;
      float up = 0.0f;
      const int64_t w1_gate_base = ((static_cast<int64_t>(e) * (2 * I) + i) * H);
      const int64_t w1_up_base = ((static_cast<int64_t>(e) * (2 * I) + I + i) * H);
      const int64_t x_base = static_cast<int64_t>(m) * H;
      for (int j = 0; j < H; ++j) {
        const float xv = x[x_base + j];
        gate += xv * w1[w1_gate_base + j];
        up += xv * w1[w1_up_base + j];
      }
      const float act = silu(gate) * up;
      expert_sum += act * w2[(static_cast<int64_t>(e) * H + h) * I + i];
    }
    acc_out += route_w * expert_sum;
  }

  out[static_cast<int64_t>(m) * H + h] = acc_out;
}

__global__ void stage1_intermediate_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w1,
    const int64_t* __restrict__ topk_ids,
    float* __restrict__ mid,
    int M,
    int H,
    int I,
    int E,
    int TOPK) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int mk = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= I || mk >= M * TOPK) {
    return;
  }
  const int m = mk / TOPK;
  const int k = mk - m * TOPK;
  const int64_t expert_id_raw = topk_ids[m * TOPK + k];
  if (expert_id_raw < 0 || expert_id_raw >= E) {
    mid[static_cast<int64_t>(mk) * I + i] = 0.0f;
    return;
  }
  const int e = static_cast<int>(expert_id_raw);
  float gate = 0.0f;
  float up = 0.0f;
  const int64_t x_base = static_cast<int64_t>(m) * H;
  const int64_t w1_gate_base = ((static_cast<int64_t>(e) * (2 * I) + i) * H);
  const int64_t w1_up_base = ((static_cast<int64_t>(e) * (2 * I) + I + i) * H);
  for (int j = 0; j < H; ++j) {
    const float xv = x[x_base + j];
    gate += xv * w1[w1_gate_base + j];
    up += xv * w1[w1_up_base + j];
  }
  mid[static_cast<int64_t>(mk) * I + i] = silu(gate) * up;
}

__global__ void stage2_output_kernel(
    const float* __restrict__ mid,
    const float* __restrict__ w2,
    const int64_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    float* __restrict__ out,
    int M,
    int H,
    int I,
    int E,
    int TOPK) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || h >= H) {
    return;
  }
  float acc_out = 0.0f;
  for (int k = 0; k < TOPK; ++k) {
    const int64_t expert_id_raw = topk_ids[m * TOPK + k];
    if (expert_id_raw < 0 || expert_id_raw >= E) {
      continue;
    }
    const int e = static_cast<int>(expert_id_raw);
    float expert_sum = 0.0f;
    const int64_t mk = static_cast<int64_t>(m) * TOPK + k;
    for (int i = 0; i < I; ++i) {
      expert_sum += mid[mk * I + i] * w2[(static_cast<int64_t>(e) * H + h) * I + i];
    }
    acc_out += topk_weights[m * TOPK + k] * expert_sum;
  }
  out[static_cast<int64_t>(m) * H + h] = acc_out;
}

}  // namespace

void fused_moe_forward_cuda(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor out) {
  const int M = static_cast<int>(x.size(0));
  const int H = static_cast<int>(x.size(1));
  const int I = static_cast<int>(w2.size(2));
  const int E = static_cast<int>(w1.size(0));
  const int TOPK = static_cast<int>(topk_ids.size(1));

  auto mid = torch::empty({M * TOPK, I}, x.options());
  const auto stream = at::cuda::getCurrentCUDAStream();

  constexpr dim3 block_stage1(16, 4);
  const dim3 grid_stage1((I + block_stage1.x - 1) / block_stage1.x,
                         (M * TOPK + block_stage1.y - 1) / block_stage1.y);
  stage1_intermediate_kernel<<<grid_stage1, block_stage1, 0, stream>>>(
      x.data_ptr<float>(),
      w1.data_ptr<float>(),
      topk_ids.data_ptr<int64_t>(),
      mid.data_ptr<float>(),
      M,
      H,
      I,
      E,
      TOPK);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr dim3 block_stage2(16, 8);
  const dim3 grid_stage2((H + block_stage2.x - 1) / block_stage2.x,
                         (M + block_stage2.y - 1) / block_stage2.y);
  stage2_output_kernel<<<grid_stage2, block_stage2, 0, stream>>>(
      mid.data_ptr<float>(),
      w2.data_ptr<float>(),
      topk_ids.data_ptr<int64_t>(),
      topk_weights.data_ptr<float>(),
      out.data_ptr<float>(),
      M,
      H,
      I,
      E,
      TOPK);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
