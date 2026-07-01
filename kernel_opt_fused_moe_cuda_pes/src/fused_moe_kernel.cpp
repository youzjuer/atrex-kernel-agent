#include <torch/extension.h>

#include <stdexcept>

void fused_moe_forward_cuda(torch::Tensor hidden_states, torch::Tensor hidden_states_scale,
                            torch::Tensor gemm1_weights, torch::Tensor gemm1_weights_scale,
                            torch::Tensor gemm1_bias, torch::Tensor gemm2_weights,
                            torch::Tensor gemm2_weights_scale, torch::Tensor gemm2_bias,
                            torch::Tensor topk_ids, torch::Tensor topk_weights,
                            torch::Tensor out, int64_t num_experts,
                            int64_t local_expert_offset, int64_t intermediate_size,
                            bool use_prepared_weight_layout);

void routing_topk_softmax_type1_cuda(torch::Tensor routing_logits, torch::Tensor topk_ids,
                                     torch::Tensor topk_weights, int64_t top_k,
                                     double routed_scaling_factor);

static void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor fused_moe_forward(torch::Tensor hidden_states, torch::Tensor hidden_states_scale,
                                torch::Tensor gemm1_weights, torch::Tensor gemm1_weights_scale,
                                torch::Tensor gemm1_bias, torch::Tensor gemm2_weights,
                                torch::Tensor gemm2_weights_scale, torch::Tensor gemm2_bias,
                                torch::Tensor topk_ids, torch::Tensor topk_weights,
                                int64_t num_experts, int64_t local_expert_offset,
                                int64_t intermediate_size, bool use_prepared_weight_layout) {
  check_tensor(hidden_states, "hidden_states");
  check_tensor(hidden_states_scale, "hidden_states_scale");
  check_tensor(gemm1_weights, "gemm1_weights");
  check_tensor(gemm1_weights_scale, "gemm1_weights_scale");
  check_tensor(gemm1_bias, "gemm1_bias");
  check_tensor(gemm2_weights, "gemm2_weights");
  check_tensor(gemm2_weights_scale, "gemm2_weights_scale");
  check_tensor(gemm2_bias, "gemm2_bias");
  check_tensor(topk_ids, "topk_ids");
  check_tensor(topk_weights, "topk_weights");

  const bool hidden_is_bf16 = hidden_states.scalar_type() == torch::kBFloat16;
  const bool hidden_is_fp4 = hidden_states.scalar_type() == torch::kUInt8;
  TORCH_CHECK(hidden_is_bf16 || hidden_is_fp4,
              "hidden_states must be bf16 [T,H] or packed fp4 uint8 [T,H/2]");
  TORCH_CHECK(hidden_states_scale.scalar_type() == torch::kFloat32,
              "hidden_states_scale must be converted to fp32 before launch");
  TORCH_CHECK(gemm1_weights.scalar_type() == torch::kUInt8,
              "gemm1_weights must be packed fp4 uint8");
  TORCH_CHECK(gemm2_weights.scalar_type() == torch::kUInt8,
              "gemm2_weights must be packed fp4 uint8");
  TORCH_CHECK(gemm1_weights_scale.scalar_type() == torch::kFloat32,
              "gemm1_weights_scale must be converted to fp32 before launch");
  TORCH_CHECK(gemm2_weights_scale.scalar_type() == torch::kFloat32,
              "gemm2_weights_scale must be converted to fp32 before launch");
  TORCH_CHECK(gemm1_bias.scalar_type() == torch::kFloat32, "gemm1_bias must be fp32");
  TORCH_CHECK(gemm2_bias.scalar_type() == torch::kFloat32, "gemm2_bias must be fp32");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt64, "topk_ids must be int64");
  TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32, "topk_weights must be fp32");

  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must have shape [T, H] or [T, H/2]");
  TORCH_CHECK(hidden_states_scale.dim() == 1 || hidden_states_scale.dim() == 2,
              "hidden_states_scale must be empty or have shape [T, H/SF]");
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
  const auto H = hidden_is_fp4 ? hidden_states.size(1) * 2 : hidden_states.size(1);
  const auto E_local = gemm1_weights.size(0);
  const auto I = intermediate_size;
  TORCH_CHECK(!hidden_is_fp4 || hidden_states.size(1) * 2 == H,
              "packed hidden_states must have shape [T, H/2]");
  TORCH_CHECK(hidden_is_bf16 || hidden_states_scale.numel() > 0,
              "hidden_states_scale is required for packed fp4 hidden_states");
  TORCH_CHECK(hidden_is_bf16 || hidden_states_scale.size(0) == T,
              "hidden_states_scale T mismatch");
  TORCH_CHECK(hidden_is_bf16 || H % hidden_states_scale.size(1) == 0,
              "invalid hidden_states_scale shape");
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

  auto out = torch::empty({T, H}, hidden_states.options().dtype(torch::kBFloat16));
  fused_moe_forward_cuda(hidden_states, hidden_states_scale, gemm1_weights,
                         gemm1_weights_scale, gemm1_bias, gemm2_weights,
                         gemm2_weights_scale, gemm2_bias, topk_ids, topk_weights, out,
                         num_experts, local_expert_offset, intermediate_size,
                         use_prepared_weight_layout);
  return out;
}

torch::Tensor fused_moe_forward_logits_type1(
    torch::Tensor routing_logits, torch::Tensor hidden_states, torch::Tensor hidden_states_scale,
    torch::Tensor gemm1_weights, torch::Tensor gemm1_weights_scale, torch::Tensor gemm1_bias,
    torch::Tensor gemm2_weights, torch::Tensor gemm2_weights_scale, torch::Tensor gemm2_bias,
    int64_t num_experts, int64_t top_k, int64_t local_expert_offset,
    int64_t intermediate_size, double routed_scaling_factor, bool use_prepared_weight_layout) {
  check_tensor(routing_logits, "routing_logits");
  TORCH_CHECK(routing_logits.dim() == 2, "routing_logits must have shape [T, num_experts]");
  TORCH_CHECK(routing_logits.scalar_type() == torch::kBFloat16 ||
                  routing_logits.scalar_type() == torch::kFloat32,
              "routing_logits must be bf16 or fp32");
  TORCH_CHECK(routing_logits.size(1) == num_experts, "routing_logits num_experts mismatch");
  TORCH_CHECK(top_k > 0 && top_k <= 16, "CUDA routing fast path requires 0 < top_k <= 16");

  auto ids_options = routing_logits.options().dtype(torch::kInt64);
  auto weights_options = routing_logits.options().dtype(torch::kFloat32);
  auto topk_ids = torch::empty({routing_logits.size(0), top_k}, ids_options);
  auto topk_weights = torch::empty({routing_logits.size(0), top_k}, weights_options);
  routing_topk_softmax_type1_cuda(routing_logits, topk_ids, topk_weights, top_k,
                                  routed_scaling_factor);
  return fused_moe_forward(hidden_states, hidden_states_scale, gemm1_weights,
                           gemm1_weights_scale, gemm1_bias, gemm2_weights,
                           gemm2_weights_scale, gemm2_bias, topk_ids, topk_weights,
                           num_experts, local_expert_offset, intermediate_size,
                           use_prepared_weight_layout);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_moe_forward,
        "FlashInfer-aligned FP4 block-scale MoE staged CUDA baseline");
  m.def("forward_logits_type1", &fused_moe_forward_logits_type1,
        "FlashInfer-aligned FP4 block-scale MoE with CUDA routing_method_type=1");
}
