#include <torch/extension.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

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

void routing_topk_softmax_type1_pack_cuda(torch::Tensor routing_logits,
                                          torch::Tensor topk_packed, int64_t top_k,
                                          double routed_scaling_factor);

void routing_metadata_from_packed_cuda(
    torch::Tensor topk_packed, torch::Tensor expanded_idx_to_permuted_idx,
    torch::Tensor permuted_idx_to_token_idx, torch::Tensor cta_idx_xy_to_batch_idx,
    torch::Tensor cta_idx_xy_to_mn_limit, torch::Tensor num_non_exiting_ctas,
    torch::Tensor total_num_padded_tokens, torch::Tensor expert_counts,
    torch::Tensor expert_padded_offsets, torch::Tensor chunk_counts,
    torch::Tensor chunk_offsets, int64_t num_experts, int64_t local_expert_offset,
    int64_t local_num_experts, int64_t tile_tokens_dim);

void pack_hidden_bmm_from_metadata_cuda(
    torch::Tensor topk_packed, torch::Tensor expanded_idx_to_permuted_idx,
    torch::Tensor expert_padded_offsets, torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale, torch::Tensor hidden_packed_bmm,
    torch::Tensor hidden_scale_bmm, int64_t num_experts, int64_t local_expert_offset,
    int64_t local_num_experts, int64_t padded_rows);

void nvfp4_block_scale_interleave_cuda(torch::Tensor scale, torch::Tensor swizzled);

void swiglu_requant_from_bmm_cuda(torch::Tensor gemm1_out, torch::Tensor expert_counts,
                                  torch::Tensor mid_packed, torch::Tensor mid_scale,
                                  torch::Tensor mid_scale_swizzled, int64_t padded_rows,
                                  int64_t intermediate_size);

void final_scatter_from_bmm_cuda(torch::Tensor gemm2_out, torch::Tensor topk_packed,
                                 torch::Tensor expanded_idx_to_permuted_idx,
                                 torch::Tensor expert_padded_offsets, torch::Tensor out,
                                 int64_t local_expert_offset, int64_t padded_rows,
                                 bool use_prepared_output_layout);

void pack_hidden_bmm_swizzled_from_metadata_cuda(
    torch::Tensor topk_packed, torch::Tensor expanded_idx_to_permuted_idx,
    torch::Tensor expert_padded_offsets, torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale, torch::Tensor hidden_packed_bmm,
    torch::Tensor hidden_scale_swizzled, int64_t num_experts,
    int64_t local_expert_offset, int64_t local_num_experts, int64_t padded_rows);

static void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static int64_t get_max_num_ctas_in_batch_dim(int64_t num_tokens, int64_t top_k,
                                             int64_t num_experts,
                                             int64_t tile_tokens_dim) {
  int64_t num_remaining_tokens = num_tokens * top_k;
  int64_t max_num_ctas = 0;
  const int64_t num_experts_filled = std::min(num_experts, num_remaining_tokens);
  max_num_ctas += num_experts_filled;
  num_remaining_tokens -= num_experts_filled;
  if (num_remaining_tokens > 0) {
    max_num_ctas += num_remaining_tokens / tile_tokens_dim;
  }
  return max_num_ctas;
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

torch::Tensor routing_pack_type1(torch::Tensor routing_logits, int64_t top_k,
                                 double routed_scaling_factor) {
  check_tensor(routing_logits, "routing_logits");
  TORCH_CHECK(routing_logits.dim() == 2, "routing_logits must have shape [T, num_experts]");
  TORCH_CHECK(routing_logits.scalar_type() == torch::kBFloat16 ||
                  routing_logits.scalar_type() == torch::kFloat32,
              "routing_logits must be bf16 or fp32");
  TORCH_CHECK(top_k > 0 && top_k <= 16, "routing_pack_type1 requires 0 < top_k <= 16");

  auto packed_options = routing_logits.options().dtype(torch::kInt32);
  auto topk_packed = torch::empty({routing_logits.size(0), top_k}, packed_options);
  routing_topk_softmax_type1_pack_cuda(routing_logits, topk_packed, top_k,
                                       routed_scaling_factor);
  return topk_packed;
}

std::vector<torch::Tensor> routing_metadata_from_packed(
    torch::Tensor topk_packed, int64_t num_experts, int64_t local_expert_offset,
    int64_t local_num_experts, int64_t tile_tokens_dim) {
  check_tensor(topk_packed, "topk_packed");
  TORCH_CHECK(topk_packed.scalar_type() == torch::kInt32, "topk_packed must be int32");
  TORCH_CHECK(topk_packed.dim() == 2, "topk_packed must have shape [T, top_k]");
  TORCH_CHECK(num_experts > 0, "num_experts must be positive");
  TORCH_CHECK(local_num_experts > 0, "local_num_experts must be positive");
  TORCH_CHECK(local_expert_offset >= 0, "local_expert_offset must be non-negative");
  TORCH_CHECK(num_experts >= local_expert_offset + local_num_experts,
              "num_experts must cover the local expert range");
  TORCH_CHECK(tile_tokens_dim > 0, "tile_tokens_dim must be positive");

  const int64_t T = topk_packed.size(0);
  const int64_t top_k = topk_packed.size(1);
  TORCH_CHECK(top_k > 0, "topk_packed top_k dimension must be positive");
  TORCH_CHECK(T <= std::numeric_limits<int32_t>::max(), "T exceeds int32 range");
  TORCH_CHECK(top_k <= std::numeric_limits<int32_t>::max(), "top_k exceeds int32 range");
  TORCH_CHECK(num_experts <= std::numeric_limits<int32_t>::max(),
              "num_experts exceeds int32 range");
  TORCH_CHECK(local_num_experts <= std::numeric_limits<int32_t>::max(),
              "local_num_experts exceeds int32 range");
  TORCH_CHECK(tile_tokens_dim <= std::numeric_limits<int32_t>::max(),
              "tile_tokens_dim exceeds int32 range");
  TORCH_CHECK(T * top_k <= std::numeric_limits<int32_t>::max(),
              "expanded token count exceeds int32 range");

  const int64_t max_num_ctas =
      get_max_num_ctas_in_batch_dim(T, top_k, num_experts, tile_tokens_dim);
  const int64_t max_num_padded_tokens = max_num_ctas * tile_tokens_dim;
  TORCH_CHECK(max_num_ctas <= std::numeric_limits<int32_t>::max(),
              "max_num_ctas exceeds int32 range");
  TORCH_CHECK(max_num_padded_tokens <= std::numeric_limits<int32_t>::max(),
              "max_num_padded_tokens exceeds int32 range");

  auto int_options = topk_packed.options().dtype(torch::kInt32);
  auto expanded_idx_to_permuted_idx = torch::empty({T, top_k}, int_options);
  auto permuted_idx_to_token_idx = torch::empty({max_num_padded_tokens}, int_options);
  auto cta_idx_xy_to_batch_idx = torch::empty({max_num_ctas}, int_options);
  auto cta_idx_xy_to_mn_limit = torch::empty({max_num_ctas}, int_options);
  auto num_non_exiting_ctas = torch::empty({1}, int_options);
  auto total_num_padded_tokens = torch::empty({1}, int_options);
  auto expert_counts = torch::empty({local_num_experts}, int_options);
  auto expert_padded_offsets = torch::empty({local_num_experts + 1}, int_options);
  constexpr int64_t rank_chunk_size = 256;
  const int64_t num_rank_chunks = (T * top_k + rank_chunk_size - 1) / rank_chunk_size;
  auto chunk_counts = torch::empty({num_rank_chunks, local_num_experts}, int_options);
  auto chunk_offsets = torch::empty({num_rank_chunks, local_num_experts}, int_options);

  routing_metadata_from_packed_cuda(
      topk_packed, expanded_idx_to_permuted_idx, permuted_idx_to_token_idx,
      cta_idx_xy_to_batch_idx, cta_idx_xy_to_mn_limit, num_non_exiting_ctas,
      total_num_padded_tokens, expert_counts, expert_padded_offsets, chunk_counts,
      chunk_offsets, num_experts, local_expert_offset, local_num_experts, tile_tokens_dim);

  return {expanded_idx_to_permuted_idx, permuted_idx_to_token_idx,
          cta_idx_xy_to_batch_idx, cta_idx_xy_to_mn_limit, num_non_exiting_ctas,
          total_num_padded_tokens, expert_counts, expert_padded_offsets};
}

std::vector<torch::Tensor> pack_hidden_bmm_from_metadata(
    torch::Tensor topk_packed, torch::Tensor expanded_idx_to_permuted_idx,
    torch::Tensor expert_padded_offsets, torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale, int64_t num_experts, int64_t local_expert_offset,
    int64_t local_num_experts, int64_t padded_rows) {
  check_tensor(topk_packed, "topk_packed");
  check_tensor(expanded_idx_to_permuted_idx, "expanded_idx_to_permuted_idx");
  check_tensor(expert_padded_offsets, "expert_padded_offsets");
  check_tensor(hidden_states, "hidden_states");
  check_tensor(hidden_states_scale, "hidden_states_scale");
  TORCH_CHECK(topk_packed.scalar_type() == torch::kInt32, "topk_packed must be int32");
  TORCH_CHECK(expanded_idx_to_permuted_idx.scalar_type() == torch::kInt32,
              "expanded_idx_to_permuted_idx must be int32");
  TORCH_CHECK(expert_padded_offsets.scalar_type() == torch::kInt32,
              "expert_padded_offsets must be int32");
  TORCH_CHECK(hidden_states.scalar_type() == torch::kUInt8,
              "hidden_states must be packed fp4 uint8");
  TORCH_CHECK(hidden_states_scale.scalar_type() == torch::kFloat8_e4m3fn ||
                  hidden_states_scale.scalar_type() == torch::kUInt8,
              "hidden_states_scale must be fp8_e4m3fn or raw uint8");
  TORCH_CHECK(topk_packed.dim() == 2, "topk_packed must have shape [T, top_k]");
  TORCH_CHECK(expanded_idx_to_permuted_idx.sizes() == topk_packed.sizes(),
              "expanded_idx_to_permuted_idx shape mismatch");
  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must have shape [T, H/2]");
  TORCH_CHECK(hidden_states_scale.dim() == 2,
              "hidden_states_scale must have shape [T, H/16]");
  TORCH_CHECK(hidden_states.size(0) == topk_packed.size(0), "hidden_states T mismatch");
  TORCH_CHECK(hidden_states_scale.size(0) == topk_packed.size(0),
              "hidden_states_scale T mismatch");
  TORCH_CHECK(hidden_states.size(1) % 16 == 0, "hidden row bytes must be 16-byte aligned");
  TORCH_CHECK((hidden_states_scale.size(1) * hidden_states_scale.element_size()) % 16 == 0,
              "hidden scale row bytes must be 16-byte aligned");
  TORCH_CHECK(num_experts > 0, "num_experts must be positive");
  TORCH_CHECK(local_num_experts > 0, "local_num_experts must be positive");
  TORCH_CHECK(local_expert_offset >= 0, "local_expert_offset must be non-negative");
  TORCH_CHECK(num_experts >= local_expert_offset + local_num_experts,
              "num_experts must cover the local expert range");
  TORCH_CHECK(padded_rows > 0, "padded_rows must be positive");
  TORCH_CHECK(expert_padded_offsets.numel() >= local_num_experts + 1,
              "expert_padded_offsets must have at least local_num_experts + 1 elements");
  TORCH_CHECK(topk_packed.numel() <= std::numeric_limits<int32_t>::max(),
              "expanded token count exceeds int32 range");

  auto hidden_packed_bmm =
      torch::empty({local_num_experts, padded_rows, hidden_states.size(1)},
                   hidden_states.options().dtype(torch::kUInt8));
  auto hidden_scale_bmm =
      torch::empty({local_num_experts, padded_rows, hidden_states_scale.size(1)},
                   hidden_states_scale.options());

  pack_hidden_bmm_from_metadata_cuda(
      topk_packed, expanded_idx_to_permuted_idx, expert_padded_offsets, hidden_states,
      hidden_states_scale, hidden_packed_bmm, hidden_scale_bmm, num_experts,
      local_expert_offset, local_num_experts, padded_rows);
  return {hidden_packed_bmm, hidden_scale_bmm};
}

torch::Tensor nvfp4_block_scale_interleave(torch::Tensor scale) {
  check_tensor(scale, "scale");
  TORCH_CHECK(scale.scalar_type() == torch::kFloat8_e4m3fn ||
                  scale.scalar_type() == torch::kUInt8,
              "scale must be fp8_e4m3fn or raw uint8");
  TORCH_CHECK(scale.dim() == 3, "scale must have shape [B, rows, K/16]");
  TORCH_CHECK(scale.element_size() == 1, "scale element size must be one byte");
  const int64_t batches = scale.size(0);
  const int64_t rows = scale.size(1);
  const int64_t scale_cols = scale.size(2);
  TORCH_CHECK(rows > 0, "scale rows must be positive");
  TORCH_CHECK(scale_cols > 0, "scale columns must be positive");
  TORCH_CHECK(scale_cols <= std::numeric_limits<int32_t>::max(),
              "scale columns exceed int32 range");
  const int64_t row_blocks = (rows + 127) / 128;
  const int64_t groups_k = (scale_cols + 3) / 4;
  const int64_t swizzled_bytes = row_blocks * groups_k * 512;
  TORCH_CHECK(swizzled_bytes <= std::numeric_limits<int32_t>::max(),
              "swizzled scale bytes per batch exceed int32 range");
  auto swizzled = torch::empty({batches, swizzled_bytes}, scale.options());
  nvfp4_block_scale_interleave_cuda(scale, swizzled);
  return swizzled;
}

std::vector<torch::Tensor> swiglu_requant_from_bmm(
    torch::Tensor gemm1_out, torch::Tensor expert_counts, int64_t padded_rows,
    int64_t intermediate_size) {
  check_tensor(gemm1_out, "gemm1_out");
  check_tensor(expert_counts, "expert_counts");
  TORCH_CHECK(gemm1_out.scalar_type() == torch::kBFloat16,
              "gemm1_out must be bf16 [E*padded_rows, 2I]");
  TORCH_CHECK(expert_counts.scalar_type() == torch::kInt32,
              "expert_counts must be int32 [E]");
  TORCH_CHECK(gemm1_out.dim() == 2, "gemm1_out must have shape [E*padded_rows, 2I]");
  TORCH_CHECK(expert_counts.dim() == 1, "expert_counts must have shape [E]");
  TORCH_CHECK(gemm1_out.is_contiguous(), "gemm1_out must be contiguous");
  TORCH_CHECK(expert_counts.is_contiguous(), "expert_counts must be contiguous");
  TORCH_CHECK(padded_rows > 0, "padded_rows must be positive");
  TORCH_CHECK(intermediate_size > 0, "intermediate_size must be positive");
  TORCH_CHECK((intermediate_size % 16) == 0, "intermediate_size must be divisible by 16");
  TORCH_CHECK(gemm1_out.size(1) == 2 * intermediate_size,
              "gemm1_out second dim must be 2I");
  TORCH_CHECK(gemm1_out.size(0) == expert_counts.size(0) * padded_rows,
              "gemm1_out first dim must be E*padded_rows");

  const int64_t local_num_experts = expert_counts.size(0);
  const int64_t scale_cols = intermediate_size / 16;
  const int64_t row_blocks = (padded_rows + 127) / 128;
  const int64_t groups_k = (scale_cols + 3) / 4;
  const int64_t swizzled_bytes = row_blocks * groups_k * 512;
  TORCH_CHECK(swizzled_bytes <= std::numeric_limits<int32_t>::max(),
              "swizzled scale bytes per expert exceed int32 range");

  auto packed_options = gemm1_out.options().dtype(torch::kUInt8);
  auto scale_options = gemm1_out.options().dtype(torch::kFloat8_e4m3fn);
  auto mid_packed =
      torch::empty({local_num_experts, padded_rows, intermediate_size / 2}, packed_options);
  auto mid_scale =
      torch::empty({local_num_experts, padded_rows, scale_cols}, scale_options);
  auto mid_scale_swizzled =
      torch::empty({local_num_experts, swizzled_bytes}, scale_options);

  swiglu_requant_from_bmm_cuda(gemm1_out, expert_counts, mid_packed, mid_scale,
                               mid_scale_swizzled, padded_rows, intermediate_size);
  return {mid_packed, mid_scale, mid_scale_swizzled};
}

torch::Tensor final_scatter_from_bmm(torch::Tensor gemm2_out, torch::Tensor topk_packed,
                                     torch::Tensor expanded_idx_to_permuted_idx,
                                     torch::Tensor expert_padded_offsets,
                                     int64_t local_expert_offset, int64_t padded_rows,
                                     bool use_prepared_output_layout) {
  check_tensor(gemm2_out, "gemm2_out");
  check_tensor(topk_packed, "topk_packed");
  check_tensor(expanded_idx_to_permuted_idx, "expanded_idx_to_permuted_idx");
  check_tensor(expert_padded_offsets, "expert_padded_offsets");
  TORCH_CHECK(gemm2_out.scalar_type() == torch::kBFloat16,
              "gemm2_out must be bf16 [E*padded_rows, H]");
  TORCH_CHECK(topk_packed.scalar_type() == torch::kInt32,
              "topk_packed must be int32 [T, top_k]");
  TORCH_CHECK(expanded_idx_to_permuted_idx.scalar_type() == torch::kInt32,
              "expanded_idx_to_permuted_idx must be int32 [T, top_k]");
  TORCH_CHECK(expert_padded_offsets.scalar_type() == torch::kInt32,
              "expert_padded_offsets must be int32 [E_local + 1]");
  TORCH_CHECK(gemm2_out.dim() == 2, "gemm2_out must have shape [E*padded_rows, H]");
  TORCH_CHECK(topk_packed.dim() == 2, "topk_packed must have shape [T, top_k]");
  TORCH_CHECK(expanded_idx_to_permuted_idx.sizes() == topk_packed.sizes(),
              "expanded_idx_to_permuted_idx shape mismatch");
  TORCH_CHECK(expert_padded_offsets.dim() == 1,
              "expert_padded_offsets must have shape [E_local + 1]");
  TORCH_CHECK(expert_padded_offsets.numel() >= 2,
              "expert_padded_offsets must have at least two entries");
  TORCH_CHECK(gemm2_out.size(1) > 0, "gemm2_out hidden dimension must be positive");
  TORCH_CHECK(topk_packed.size(1) > 0, "top_k must be positive");
  TORCH_CHECK(topk_packed.size(1) <= 16, "top_k must be <= 16");
  TORCH_CHECK(local_expert_offset >= 0, "local_expert_offset must be non-negative");
  TORCH_CHECK(padded_rows > 0, "padded_rows must be positive");
  const int64_t local_num_experts = expert_padded_offsets.numel() - 1;
  TORCH_CHECK(gemm2_out.size(0) == local_num_experts * padded_rows,
              "gemm2_out first dimension must be E_local*padded_rows");
  TORCH_CHECK(topk_packed.size(0) <= std::numeric_limits<int32_t>::max(),
              "T exceeds int32 range");
  TORCH_CHECK(topk_packed.size(1) <= std::numeric_limits<int32_t>::max(),
              "top_k exceeds int32 range");
  TORCH_CHECK(gemm2_out.size(0) <= std::numeric_limits<int32_t>::max(),
              "gemm2_out rows exceed int32 range");
  TORCH_CHECK(gemm2_out.size(1) <= std::numeric_limits<int32_t>::max(),
              "hidden size exceeds int32 range");
  TORCH_CHECK(local_expert_offset <= std::numeric_limits<int32_t>::max(),
              "local_expert_offset exceeds int32 range");
  TORCH_CHECK(local_num_experts <= std::numeric_limits<int32_t>::max(),
              "local_num_experts exceeds int32 range");
  TORCH_CHECK(padded_rows <= std::numeric_limits<int32_t>::max(),
              "padded_rows exceeds int32 range");

  auto out = torch::empty({topk_packed.size(0), gemm2_out.size(1)}, gemm2_out.options());
  final_scatter_from_bmm_cuda(gemm2_out, topk_packed, expanded_idx_to_permuted_idx,
                              expert_padded_offsets, out, local_expert_offset,
                              padded_rows, use_prepared_output_layout);
  return out;
}

std::vector<torch::Tensor> pack_hidden_bmm_swizzled_from_metadata(
    torch::Tensor topk_packed, torch::Tensor expanded_idx_to_permuted_idx,
    torch::Tensor expert_padded_offsets, torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale, int64_t num_experts, int64_t local_expert_offset,
    int64_t local_num_experts, int64_t padded_rows) {
  check_tensor(topk_packed, "topk_packed");
  check_tensor(expanded_idx_to_permuted_idx, "expanded_idx_to_permuted_idx");
  check_tensor(expert_padded_offsets, "expert_padded_offsets");
  check_tensor(hidden_states, "hidden_states");
  check_tensor(hidden_states_scale, "hidden_states_scale");
  TORCH_CHECK(topk_packed.scalar_type() == torch::kInt32, "topk_packed must be int32");
  TORCH_CHECK(expanded_idx_to_permuted_idx.scalar_type() == torch::kInt32,
              "expanded_idx_to_permuted_idx must be int32");
  TORCH_CHECK(expert_padded_offsets.scalar_type() == torch::kInt32,
              "expert_padded_offsets must be int32");
  TORCH_CHECK(hidden_states.scalar_type() == torch::kUInt8,
              "hidden_states must be packed fp4 uint8");
  TORCH_CHECK(hidden_states_scale.scalar_type() == torch::kFloat8_e4m3fn ||
                  hidden_states_scale.scalar_type() == torch::kUInt8,
              "hidden_states_scale must be fp8_e4m3fn or raw uint8");
  TORCH_CHECK(hidden_states_scale.element_size() == 1,
              "hidden_states_scale element size must be one byte");
  TORCH_CHECK(topk_packed.dim() == 2, "topk_packed must have shape [T, top_k]");
  TORCH_CHECK(expanded_idx_to_permuted_idx.sizes() == topk_packed.sizes(),
              "expanded_idx_to_permuted_idx shape mismatch");
  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must have shape [T, H/2]");
  TORCH_CHECK(hidden_states_scale.dim() == 2,
              "hidden_states_scale must have shape [T, H/16]");
  TORCH_CHECK(hidden_states.size(0) == topk_packed.size(0), "hidden_states T mismatch");
  TORCH_CHECK(hidden_states_scale.size(0) == topk_packed.size(0),
              "hidden_states_scale T mismatch");
  TORCH_CHECK(hidden_states.size(1) % 16 == 0, "hidden row bytes must be 16-byte aligned");
  TORCH_CHECK(hidden_states_scale.size(1) % 16 == 0,
              "hidden scale row bytes must be 16-byte aligned");
  TORCH_CHECK(num_experts > 0, "num_experts must be positive");
  TORCH_CHECK(local_num_experts > 0, "local_num_experts must be positive");
  TORCH_CHECK(local_expert_offset >= 0, "local_expert_offset must be non-negative");
  TORCH_CHECK(num_experts >= local_expert_offset + local_num_experts,
              "num_experts must cover the local expert range");
  TORCH_CHECK(padded_rows > 0, "padded_rows must be positive");
  TORCH_CHECK(expert_padded_offsets.numel() >= local_num_experts + 1,
              "expert_padded_offsets must have at least local_num_experts + 1 elements");
  TORCH_CHECK(topk_packed.numel() <= std::numeric_limits<int32_t>::max(),
              "expanded token count exceeds int32 range");

  const int64_t scale_cols = hidden_states_scale.size(1);
  const int64_t row_blocks = (padded_rows + 127) / 128;
  const int64_t groups_k = (scale_cols + 3) / 4;
  const int64_t swizzled_bytes = row_blocks * groups_k * 512;
  TORCH_CHECK(swizzled_bytes <= std::numeric_limits<int32_t>::max(),
              "swizzled scale bytes per expert exceed int32 range");

  auto hidden_packed_bmm =
      torch::empty({local_num_experts, padded_rows, hidden_states.size(1)},
                   hidden_states.options().dtype(torch::kUInt8));
  auto hidden_scale_swizzled =
      torch::empty({local_num_experts, swizzled_bytes}, hidden_states_scale.options());

  pack_hidden_bmm_swizzled_from_metadata_cuda(
      topk_packed, expanded_idx_to_permuted_idx, expert_padded_offsets, hidden_states,
      hidden_states_scale, hidden_packed_bmm, hidden_scale_swizzled, num_experts,
      local_expert_offset, local_num_experts, padded_rows);
  return {hidden_packed_bmm, hidden_scale_swizzled};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_moe_forward,
        "FlashInfer-aligned FP4 block-scale MoE staged CUDA baseline");
  m.def("forward_logits_type1", &fused_moe_forward_logits_type1,
        "FlashInfer-aligned FP4 block-scale MoE with CUDA routing_method_type=1");
  m.def("routing_pack_type1", &routing_pack_type1,
        "Pack routing_method_type=1 TopK as TRT-LLM PackedScoreIdx<bf16>");
  m.def("routing_metadata_from_packed", &routing_metadata_from_packed,
        "Build TRT-LLM MoE grouped-GEMM routing metadata from PackedScoreIdx<bf16>");
  m.def("pack_hidden_bmm_from_metadata", &pack_hidden_bmm_from_metadata,
        "Pack G11 hidden FP4 rows into expert-major BMM layout using routing metadata");
  m.def("nvfp4_block_scale_interleave", &nvfp4_block_scale_interleave,
        "Interleave linear NVFP4 block scales into FlashInfer/SM100 BMM layout");
  m.def("swiglu_requant_from_bmm", &swiglu_requant_from_bmm,
        "Apply prepared-row-aware SwiGLU to GEMM1 BMM output and requantize to NVFP4");
  m.def("final_scatter_from_bmm", &final_scatter_from_bmm,
        "Apply packed bf16 top-k weights and scatter GEMM2 BMM output to [T, H]");
  m.def("pack_hidden_bmm_swizzled_from_metadata", &pack_hidden_bmm_swizzled_from_metadata,
        "Pack G11 hidden FP4 rows and swizzled scales using routing metadata");
}
