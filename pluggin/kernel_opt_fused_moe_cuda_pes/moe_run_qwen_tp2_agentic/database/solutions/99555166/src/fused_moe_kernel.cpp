#include <torch/extension.h>

#include <stdexcept>

void fused_moe_forward_cuda(torch::Tensor hidden_states, torch::Tensor gemm1_weights,
                            torch::Tensor gemm1_weights_scale, torch::Tensor gemm1_bias,
                            torch::Tensor gemm2_weights, torch::Tensor gemm2_weights_scale,
                            torch::Tensor gemm2_bias, torch::Tensor topk_ids,
                            torch::Tensor topk_weights, torch::Tensor out,
                            int64_t num_experts, int64_t local_expert_offset,
                            int64_t intermediate_size);

static void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor fused_moe_forward(torch::Tensor hidden_states, torch::Tensor gemm1_weights,
                                torch::Tensor gemm1_weights_scale, torch::Tensor gemm1_bias,
                                torch::Tensor gemm2_weights, torch::Tensor gemm2_weights_scale,
                                torch::Tensor gemm2_bias, torch::Tensor topk_ids,
                                torch::Tensor topk_weights, int64_t num_experts,
                                int64_t local_expert_offset, int64_t intermediate_size) {
  check_tensor(hidden_states, "hidden_states");
  check_tensor(gemm1_weights, "gemm1_weights");
  check_tensor(gemm1_weights_scale, "gemm1_weights_scale");
  check_tensor(gemm1_bias, "gemm1_bias");
  check_tensor(gemm2_weights, "gemm2_weights");
  check_tensor(gemm2_weights_scale, "gemm2_weights_scale");
  check_tensor(gemm2_bias, "gemm2_bias");
  check_tensor(topk_ids, "topk_ids");
  check_tensor(topk_weights, "topk_weights");

  TORCH_CHECK(hidden_states.scalar_type() == torch::kBFloat16, "hidden_states must be bf16");
  TORCH_CHECK(gemm1_weights.scalar_type() == torch::kUInt8,
              "gemm1_weights must be packed fp4 uint8");
  TORCH_CHECK(gemm2_weights.scalar_type() == torch::kUInt8,
              "gemm2_weights must be packed fp4 uint8");
  TORCH_CHECK(gemm1_weights_scale.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "gemm1_weights_scale must be float8_e4m3fn");
  TORCH_CHECK(gemm2_weights_scale.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "gemm2_weights_scale must be float8_e4m3fn");
  TORCH_CHECK(gemm1_bias.scalar_type() == torch::kFloat32, "gemm1_bias must be fp32");
  TORCH_CHECK(gemm2_bias.scalar_type() == torch::kFloat32, "gemm2_bias must be fp32");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt64, "topk_ids must be int64");
  TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32, "topk_weights must be fp32");

  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must have shape [T, H]");
  TORCH_CHECK(gemm1_weights.dim() == 3,
              "gemm1_weights must have shape [E_local, 2I, H/2]");
  TORCH_CHECK(gemm2_weights.dim() == 3,
              "gemm2_weights must have shape [E_local, H, I/2]");
  TORCH_CHECK(gemm1_weights_scale.dim() == 3,
              "gemm1_weights_scale must have shape [E_local, 2I, H/SF]");
  TORCH_CHECK(gemm2_weights_scale.dim() == 3,
              "gemm2_weights_scale must have shape [E_local, H, I/SF]");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must have shape [T, top_k]");
  TORCH_CHECK(topk_weights.sizes() == topk_ids.sizes(), "topk_weights shape mismatch");

  const auto T = hidden_states.size(0);
  const auto H = hidden_states.size(1);
  const auto E_local = gemm1_weights.size(0);
  const auto I = intermediate_size;
  TORCH_CHECK(gemm1_weights.size(1) == 2 * I, "gemm1_weights second dim must be 2I");
  TORCH_CHECK(gemm1_weights.size(2) * 2 == H, "gemm1_weights packed H mismatch");
  TORCH_CHECK(gemm2_weights.size(0) == E_local, "gemm2_weights E_local mismatch");
  TORCH_CHECK(gemm2_weights.size(1) == H, "gemm2_weights H mismatch");
  TORCH_CHECK(gemm2_weights.size(2) * 2 == I, "gemm2_weights packed I mismatch");
  TORCH_CHECK(gemm1_weights_scale.size(0) == E_local, "gemm1 scale E mismatch");
  TORCH_CHECK(gemm1_weights_scale.size(1) == 2 * I, "gemm1 scale row mismatch");
  TORCH_CHECK(gemm2_weights_scale.size(0) == E_local, "gemm2 scale E mismatch");
  TORCH_CHECK(gemm2_weights_scale.size(1) == H, "gemm2 scale row mismatch");
  TORCH_CHECK(H % gemm1_weights_scale.size(2) == 0, "invalid gemm1 scale shape");
  TORCH_CHECK(I % gemm2_weights_scale.size(2) == 0, "invalid gemm2 scale shape");
  TORCH_CHECK(topk_ids.size(0) == T, "topk_ids T mismatch");
  TORCH_CHECK(gemm1_bias.numel() == 0 || gemm1_bias.numel() == E_local * 2 * I,
              "gemm1_bias must be empty or [E_local, 2I]");
  TORCH_CHECK(gemm2_bias.numel() == 0 || gemm2_bias.numel() == E_local * H,
              "gemm2_bias must be empty or [E_local, H]");
  TORCH_CHECK(num_experts >= E_local + local_expert_offset,
              "num_experts must cover local expert range");

  auto out = torch::empty({T, H}, hidden_states.options());
  fused_moe_forward_cuda(hidden_states, gemm1_weights, gemm1_weights_scale, gemm1_bias,
                         gemm2_weights, gemm2_weights_scale, gemm2_bias, topk_ids,
                         topk_weights, out, num_experts, local_expert_offset,
                         intermediate_size);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_moe_forward,
        "FlashInfer-aligned FP4 block-scale MoE staged CUDA baseline");
}
