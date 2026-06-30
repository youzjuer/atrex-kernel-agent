#include <torch/extension.h>

#include <stdexcept>

void fused_moe_forward_cuda(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    torch::Tensor out);

static void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor fused_moe_forward(
    torch::Tensor x,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights) {
  check_tensor(x, "x");
  check_tensor(w1, "w1");
  check_tensor(w2, "w2");
  check_tensor(topk_ids, "topk_ids");
  check_tensor(topk_weights, "topk_weights");

  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
  TORCH_CHECK(w1.scalar_type() == torch::kFloat32, "w1 must be fp32");
  TORCH_CHECK(w2.scalar_type() == torch::kFloat32, "w2 must be fp32");
  TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32, "topk_weights must be fp32");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt64, "topk_ids must be int64");

  TORCH_CHECK(x.dim() == 2, "x must have shape [M, H]");
  TORCH_CHECK(w1.dim() == 3, "w1 must have shape [E, 2I, H]");
  TORCH_CHECK(w2.dim() == 3, "w2 must have shape [E, H, I]");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must have shape [M, TOPK]");
  TORCH_CHECK(topk_weights.sizes() == topk_ids.sizes(), "topk_weights shape mismatch");

  const auto M = x.size(0);
  const auto H = x.size(1);
  const auto E = w1.size(0);
  const auto twoI = w1.size(1);
  TORCH_CHECK(twoI % 2 == 0, "w1 second dimension must be 2 * I");
  const auto I = twoI / 2;
  TORCH_CHECK(w1.size(2) == H, "w1 H mismatch");
  TORCH_CHECK(w2.size(0) == E, "w2 E mismatch");
  TORCH_CHECK(w2.size(1) == H, "w2 H mismatch");
  TORCH_CHECK(w2.size(2) == I, "w2 I mismatch");
  TORCH_CHECK(topk_ids.size(0) == M, "topk_ids M mismatch");

  auto out = torch::empty({M, H}, x.options());
  fused_moe_forward_cuda(x, w1, w2, topk_ids, topk_weights, out);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_moe_forward, "Fused MoE forward baseline (CUDA)");
}
