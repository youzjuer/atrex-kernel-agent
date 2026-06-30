# Baseline Report

## Environment

- branch: `dev_ycb`
- GPU: NVIDIA L20D
- CUDA: `/usr/local/cuda`
- PyTorch CUDA: available
- FlashInfer installed locally: `0.6.9`

## Operator Contract

- target API: `flashinfer.trtllm_fp4_block_scale_moe`
- application profile: `Qwen3_5-Plus_prefill_TP2`
- Qwen metadata: `num_experts=512`, `top_k=10`, `hidden_size=4096`,
  `intermediate_size=1024`, TP2 local expert shard `local_num_experts=256`
- current smoke preset: `H=128`, `I=64`, `E_global=512`, `top_k=10`,
  `E_local=16`

## Baseline

- Candidate: `kernel.py`
- CUDA sources: `src/fused_moe_kernel.cpp`, `src/fused_moe_kernel.cu`
- Reference: `reference.py`
- Evaluator: `test_kernel.py`

The baseline now accepts the FlashInfer signature, computes routing from
`routing_logits`, consumes packed e2m1 FP4 weights and fp8 block scales, applies
TRT-LLM SwiGLU semantics `silu(X2) * X1`, and returns finalized bf16 output.

## Validation

Commands:

```bash
python test_kernel.py --mode correctness --preset smoke --tokens 2,4 \
  --json-out profiles/flashinfer_aligned_smoke.json

python test_kernel.py --mode correctness --preset qwen_micro --tokens 1 \
  --json-out profiles/flashinfer_aligned_qwen_micro.json

python test_kernel.py --mode profile --preset smoke --tokens 2 --warmup 2 --rep 5 \
  --json-out profiles/flashinfer_aligned_profile_smoke.json
```

Results:

```text
smoke correctness: PASS, max_abs=0.0, max_rel=0.0
qwen_micro correctness: PASS, max_abs=0.0, max_rel=0.0
smoke profile: PASS, candidate_us=26201.7288, reference_us=2376.8640
```

## Next PES Directions

- Replace the scalar per-output CUDA path with grouped token/expert scheduling.
- Dequantize FP4 tiles once per CTA and reuse across gate/up and GEMM2 work.
- Fuse routing output layout directly into grouped GEMM inputs.
- Add a production-shape allocation/interface probe separate from scalar correctness tests.
