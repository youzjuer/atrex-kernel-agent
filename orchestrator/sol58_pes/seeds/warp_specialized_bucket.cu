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

__global__ void count_assignments(
    const int* __restrict__ keys,
    int* __restrict__ offsets,
    int n) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < n;
         index += blockDim.x * gridDim.x) {
        atomicAdd(offsets + keys[index] + 1, 1);
    }
}

__global__ void prefix_counts(int* offsets) {
    __shared__ int warp_totals[8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int value = offsets[tid + 1];
    int inclusive = warp_inclusive(value);
    if (lane == 31) warp_totals[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        int total = lane < 8 ? warp_totals[lane] : 0;
        total = warp_inclusive(total);
        if (lane < 8) warp_totals[lane] = total;
    }
    __syncthreads();
    const int prior = warp == 0 ? 0 : warp_totals[warp - 1];
    if (tid == 0) offsets[0] = 0;
    offsets[tid + 1] = inclusive + prior;
}

__global__ void warp_specialized_stable_scatter(
    const int* __restrict__ keys,
    int* __restrict__ output,
    const int* __restrict__ offsets,
    int n) {
    __shared__ int warp_totals[8];
    __shared__ int tile_total;
    const int bucket = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const bool producer_warp = warp_id == 0;
    const bool consumer_warp = warp_id > 0;
    int output_base = offsets[bucket];
    for (int tile = 0; tile < n; tile += 256) {
        const int index = tile + tid;
        int selected = index < n && keys[index] == bucket;
        if (producer_warp || consumer_warp) {
            const int rank = block_exclusive(selected, warp_totals, &tile_total);
            if (selected) output[output_base + rank] = index;
        }
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
    count_assignments<<<128, 256, 0, stream>>>(
        topk_idx.data_ptr<int>(), expert_offsets.data_ptr<int>(), n);
    CUDA_CHECK(cudaGetLastError());
    prefix_counts<<<1, 256, 0, stream>>>(expert_offsets.data_ptr<int>());
    CUDA_CHECK(cudaGetLastError());
    warp_specialized_stable_scatter<<<256, 256, 0, stream>>>(
        topk_idx.data_ptr<int>(),
        sorted_token_indices.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        n);
    CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "SOL58 warp-specialized stable bucket seed");
}
