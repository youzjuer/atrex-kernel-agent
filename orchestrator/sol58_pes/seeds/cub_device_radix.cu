#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <unordered_map>

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

__global__ void initialize_pairs_and_histogram(
    const int* __restrict__ keys,
    int* __restrict__ values,
    int* __restrict__ expert_offsets,
    int n) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < n;
         index += blockDim.x * gridDim.x) {
        values[index] = index;
        atomicAdd(expert_offsets + keys[index] + 1, 1);
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
    const int prior_warps = warp == 0 ? 0 : warp_totals[warp - 1];
    if (tid == 0) expert_offsets[0] = 0;
    expert_offsets[tid + 1] = inclusive + prior_warps;
}

struct RadixScratch {
    torch::Tensor keys_out;
    torch::Tensor values_in;
    torch::Tensor temporary;
    size_t temporary_bytes = 0;
};

std::mutex scratch_mutex;
std::unordered_map<std::uint64_t, RadixScratch> scratch_cache;

std::uint64_t cache_key(int device, int n) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(device)) << 32) |
           static_cast<std::uint32_t>(n);
}

RadixScratch& radix_scratch(
    const torch::Tensor& topk_idx,
    torch::Tensor& sorted_token_indices,
    cudaStream_t stream) {
    const int device = topk_idx.get_device();
    const int n = static_cast<int>(topk_idx.numel());
    const std::uint64_t key = cache_key(device, n);
    std::lock_guard<std::mutex> lock(scratch_mutex);
    auto found = scratch_cache.find(key);
    if (found != scratch_cache.end()) return found->second;

    RadixScratch scratch;
    scratch.keys_out = torch::empty({n}, topk_idx.options());
    scratch.values_in = torch::empty({n}, topk_idx.options());
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr,
        scratch.temporary_bytes,
        topk_idx.data_ptr<int>(),
        scratch.keys_out.data_ptr<int>(),
        scratch.values_in.data_ptr<int>(),
        sorted_token_indices.data_ptr<int>(),
        n,
        0,
        8,
        stream));
    scratch.temporary = torch::empty(
        {static_cast<int64_t>(scratch.temporary_bytes)},
        topk_idx.options().dtype(torch::kUInt8));
    return scratch_cache.emplace(key, std::move(scratch)).first->second;
}

}  // namespace

void run(
    torch::Tensor topk_idx,
    torch::Tensor sorted_token_indices,
    torch::Tensor expert_offsets) {
    const at::cuda::CUDAGuard guard(topk_idx.device());
    const int n = static_cast<int>(topk_idx.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    RadixScratch& scratch = radix_scratch(topk_idx, sorted_token_indices, stream);

    CUDA_CHECK(cudaMemsetAsync(
        expert_offsets.data_ptr<int>(), 0, 257 * sizeof(int), stream));
    const int blocks = std::min(256, (n + 255) / 256);
    initialize_pairs_and_histogram<<<blocks, 256, 0, stream>>>(
        topk_idx.data_ptr<int>(),
        scratch.values_in.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        n);
    CUDA_CHECK(cudaGetLastError());
    prefix_expert_counts<<<1, 256, 0, stream>>>(expert_offsets.data_ptr<int>());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        scratch.temporary.data_ptr(),
        scratch.temporary_bytes,
        topk_idx.data_ptr<int>(),
        scratch.keys_out.data_ptr<int>(),
        scratch.values_in.data_ptr<int>(),
        sorted_token_indices.data_ptr<int>(),
        n,
        0,
        8,
        stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "SOL58 CUB stable 8-bit radix architecture seed");
}
