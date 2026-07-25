#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace {

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t err = (expr);                                               \
        TORCH_CHECK(err == cudaSuccess, #expr " failed: ",                    \
                    cudaGetErrorString(err));                                   \
    } while (0)

__device__ __forceinline__ int warp_inclusive(int value) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, value, offset);
        if (lane >= offset) value += other;
    }
    return value;
}

__device__ __forceinline__ int block_exclusive(
    int predicate,
    int* warp_totals,
    int* block_total) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int inclusive = warp_inclusive(predicate);
    if (lane == 31) warp_totals[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        int value = lane < 8 ? warp_totals[lane] : 0;
        value = warp_inclusive(value);
        if (lane < 8) warp_totals[lane] = value;
    }
    __syncthreads();
    const int prior = warp == 0 ? 0 : warp_totals[warp - 1];
    if (tid == 255) *block_total = inclusive + prior;
    __syncthreads();
    return inclusive + prior - predicate;
}

__global__ void count_experts(
    const int* __restrict__ topk,
    int* __restrict__ expert_offsets,
    int n) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < n;
         index += blockDim.x * gridDim.x) {
        atomicAdd(expert_offsets + topk[index] + 1, 1);
    }
}

__global__ void prefix_expert_counts(int* expert_offsets) {
    __shared__ int warp_totals[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int count = expert_offsets[tid + 1];
    int inclusive = warp_inclusive(count);
    if (lane == 31) warp_totals[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        int value = lane < 8 ? warp_totals[lane] : 0;
        value = warp_inclusive(value);
        if (lane < 8) warp_totals[lane] = value;
    }
    __syncthreads();
    const int prior = warp == 0 ? 0 : warp_totals[warp - 1];
    if (tid == 0) expert_offsets[0] = 0;
    expert_offsets[tid + 1] = inclusive + prior;
}

__global__ void expert_parallel_stable_scan(
    const int* __restrict__ topk,
    int* __restrict__ sorted_token_indices,
    const int* __restrict__ expert_offsets,
    int n) {
    __shared__ int warp_totals[8];
    __shared__ int tile_total;
    const int expert = blockIdx.x;
    const int tid = threadIdx.x;
    int output_base = expert_offsets[expert];

    for (int tile = 0; tile < n; tile += 256) {
        const int index = tile + tid;
        const int selected = index < n && topk[index] == expert;
        const int rank = block_exclusive(selected, warp_totals, &tile_total);
        if (selected) sorted_token_indices[output_base + rank] = index;
        __syncthreads();
        output_base += tile_total;
        __syncthreads();
    }
}

}  // namespace

void run(
    torch::Tensor topk_idx,
    torch::Tensor sorted_token_indices,
    torch::Tensor expert_offsets) {
    const at::cuda::CUDAGuard guard(topk_idx.device());
    const int n = static_cast<int>(topk_idx.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CUDA_CHECK(cudaMemsetAsync(
        expert_offsets.data_ptr<int>(), 0, 257 * sizeof(int), stream));
    count_experts<<<128, 256, 0, stream>>>(
        topk_idx.data_ptr<int>(), expert_offsets.data_ptr<int>(), n);
    CUDA_CHECK(cudaGetLastError());
    prefix_expert_counts<<<1, 256, 0, stream>>>(expert_offsets.data_ptr<int>());
    CUDA_CHECK(cudaGetLastError());
    expert_parallel_stable_scan<<<256, 256, 0, stream>>>(
        topk_idx.data_ptr<int>(),
        sorted_token_indices.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        n);
    CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "SOL58 expert-parallel stable scan architecture seed");
}
