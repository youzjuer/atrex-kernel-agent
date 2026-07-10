#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <string>
#include <unordered_map>

namespace {

constexpr int kExperts = 256;
constexpr int kTile = 256;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxBlocks = 256;

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t err = (expr);                                             \
        TORCH_CHECK(err == cudaSuccess, #expr " failed: ", cudaGetErrorString(err)); \
    } while (0)

__device__ __forceinline__ int warp_inclusive_scan(int v, int lane) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int add = __shfl_up_sync(0xffffffffu, v, offset);
        if (lane >= offset) {
            v += add;
        }
    }
    return v;
}

__device__ __forceinline__ int block_inclusive_scan_256(
    int v,
    int tid,
    int* warp_totals) {
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int scan = warp_inclusive_scan(v, lane);

    if (lane == 31) {
        warp_totals[warp] = scan;
    }
    __syncthreads();

    if (warp == 0) {
        int wt = lane < kWarps ? warp_totals[lane] : 0;
        int wp = warp_inclusive_scan(wt, lane);
        if (lane < kWarps) {
            warp_totals[lane] = wp - wt;
        }
    }
    __syncthreads();

    return scan + warp_totals[warp];
}

__global__ __launch_bounds__(kThreads, 2)
void count_tiles_kernel(
    const int* __restrict__ topk,
    int* __restrict__ counts,
    int n) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * kTile + tid;
    __shared__ int hist[kExperts];

    hist[tid] = 0;
    __syncthreads();

    if (idx < n) {
        atomicAdd(&hist[topk[idx]], 1);
    }
    __syncthreads();

    counts[blockIdx.x * kExperts + tid] = hist[tid];
}

__global__ __launch_bounds__(kThreads, 2)
void prefix_counts_kernel(
    const int* __restrict__ counts,
    int* __restrict__ block_offsets,
    int* __restrict__ expert_offsets,
    int n_blocks) {
    const int expert = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int warp_totals[kWarps];

    int c = tid < n_blocks ? counts[tid * kExperts + expert] : 0;
    int scan = block_inclusive_scan_256(c, tid, warp_totals);

    if (tid < n_blocks) {
        block_offsets[tid * kExperts + expert] = scan - c;
    }
    if (tid == kThreads - 1) {
        expert_offsets[expert + 1] = scan;
    }
}

__global__ __launch_bounds__(kThreads, 1)
void expert_offsets_kernel(int* __restrict__ expert_offsets) {
    const int tid = threadIdx.x;
    __shared__ int warp_totals[kWarps];

    int total = expert_offsets[tid + 1];
    int scan = block_inclusive_scan_256(total, tid, warp_totals);
    if (tid == 0) {
        expert_offsets[0] = 0;
    }
    expert_offsets[tid + 1] = scan;
}

__global__ __launch_bounds__(kThreads, 2)
void scatter_tiles_kernel(
    const int* __restrict__ topk,
    int* __restrict__ sorted_token_indices,
    const int* __restrict__ block_offsets,
    const int* __restrict__ expert_offsets,
    int n) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * kTile + tid;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned int lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
    __shared__ int warp_hist[kWarps * kExperts];

    for (int i = tid; i < kWarps * kExperts; i += kThreads) {
        warp_hist[i] = 0;
    }
    __syncthreads();

    int expert = 0;
    bool active = idx < n;
    if (active) {
        expert = topk[idx];
        atomicAdd(&warp_hist[warp * kExperts + expert], 1);
    }
    __syncthreads();

    const unsigned int same_warp = __match_any_sync(0xffffffffu, active ? expert : -1);
    const int rank_in_warp = active ? __popc(same_warp & lane_mask) : 0;

    int prior_warps = 0;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        if (w < warp && active) {
            prior_warps += warp_hist[w * kExperts + expert];
        }
    }

    if (active) {
        int out = expert_offsets[expert] +
                  block_offsets[blockIdx.x * kExperts + expert] +
                  prior_warps + rank_in_warp;
        sorted_token_indices[out] = idx;
    }
}

torch::Tensor& scratch_tensor(int index, torch::Device device) {
    static torch::Tensor scratch[2];
    static int scratch_device[2] = {-1, -1};
    const int dev = device.index();
    if (!scratch[index].defined() || scratch_device[index] != dev) {
        c10::cuda::CUDAGuard guard(device);
        auto opts = torch::TensorOptions().device(device).dtype(torch::kInt32);
        scratch[index] = torch::empty({kMaxBlocks * kExperts}, opts);
        scratch_device[index] = dev;
    }
    return scratch[index];
}

struct GraphPlan {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t count_node = nullptr;
    cudaGraphNode_t prefix_node = nullptr;
    cudaGraphNode_t expert_node = nullptr;
    cudaGraphNode_t scatter_node = nullptr;
    int n = 0;
    int n_blocks = 0;
};

GraphPlan& graph_plan(int device_index, int n, int n_blocks) {
    static std::unordered_map<long long, GraphPlan> plans;
    long long key = (static_cast<long long>(device_index) << 32) | static_cast<unsigned int>(n);
    auto it = plans.find(key);
    if (it != plans.end()) {
        return it->second;
    }

    GraphPlan plan;
    plan.n = n;
    plan.n_blocks = n_blocks;

    const int* dummy_topk = nullptr;
    const int* dummy_counts_const = nullptr;
    const int* dummy_block_offsets_const = nullptr;
    const int* dummy_expert_offsets_const = nullptr;
    int* dummy_counts = nullptr;
    int* dummy_block_offsets = nullptr;
    int* dummy_sorted = nullptr;
    int* dummy_expert_offsets = nullptr;
    int n_arg = n;
    int n_blocks_arg = n_blocks;

    CUDA_CHECK(cudaGraphCreate(&plan.graph, 0));

    cudaKernelNodeParams count_params{};
    void* count_args[] = {&dummy_topk, &dummy_counts, &n_arg};
    count_params.func = reinterpret_cast<void*>(count_tiles_kernel);
    count_params.gridDim = dim3(n_blocks);
    count_params.blockDim = dim3(kThreads);
    count_params.sharedMemBytes = 0;
    count_params.kernelParams = count_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&plan.count_node, plan.graph, nullptr, 0, &count_params));

    dummy_counts_const = dummy_counts;
    cudaKernelNodeParams prefix_params{};
    void* prefix_args[] = {&dummy_counts_const, &dummy_block_offsets, &dummy_expert_offsets, &n_blocks_arg};
    prefix_params.func = reinterpret_cast<void*>(prefix_counts_kernel);
    prefix_params.gridDim = dim3(kExperts);
    prefix_params.blockDim = dim3(kThreads);
    prefix_params.sharedMemBytes = 0;
    prefix_params.kernelParams = prefix_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&plan.prefix_node, plan.graph, &plan.count_node, 1, &prefix_params));

    cudaKernelNodeParams expert_params{};
    void* expert_args[] = {&dummy_expert_offsets};
    expert_params.func = reinterpret_cast<void*>(expert_offsets_kernel);
    expert_params.gridDim = dim3(1);
    expert_params.blockDim = dim3(kThreads);
    expert_params.sharedMemBytes = 0;
    expert_params.kernelParams = expert_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&plan.expert_node, plan.graph, &plan.prefix_node, 1, &expert_params));

    dummy_block_offsets_const = dummy_block_offsets;
    dummy_expert_offsets_const = dummy_expert_offsets;
    cudaKernelNodeParams scatter_params{};
    void* scatter_args[] = {
        &dummy_topk,
        &dummy_sorted,
        &dummy_block_offsets_const,
        &dummy_expert_offsets_const,
        &n_arg,
    };
    scatter_params.func = reinterpret_cast<void*>(scatter_tiles_kernel);
    scatter_params.gridDim = dim3(n_blocks);
    scatter_params.blockDim = dim3(kThreads);
    scatter_params.sharedMemBytes = 0;
    scatter_params.kernelParams = scatter_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&plan.scatter_node, plan.graph, &plan.expert_node, 1, &scatter_params));
    CUDA_CHECK(cudaGraphInstantiate(&plan.exec, plan.graph, 0));

    auto [inserted, _] = plans.emplace(key, plan);
    return inserted->second;
}

void launch_graph(
    GraphPlan& plan,
    const int* topk,
    int* counts,
    int* block_offsets,
    int* sorted_token_indices,
    int* expert_offsets,
    cudaStream_t stream) {
    const int* topk_arg = topk;
    const int* counts_const_arg = counts;
    const int* block_offsets_const_arg = block_offsets;
    const int* expert_offsets_const_arg = expert_offsets;
    int* counts_arg = counts;
    int* block_offsets_arg = block_offsets;
    int* sorted_arg = sorted_token_indices;
    int* expert_offsets_arg = expert_offsets;
    int n_arg = plan.n;
    int n_blocks_arg = plan.n_blocks;

    cudaKernelNodeParams count_params{};
    void* count_args[] = {&topk_arg, &counts_arg, &n_arg};
    count_params.func = reinterpret_cast<void*>(count_tiles_kernel);
    count_params.gridDim = dim3(plan.n_blocks);
    count_params.blockDim = dim3(kThreads);
    count_params.sharedMemBytes = 0;
    count_params.kernelParams = count_args;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(plan.exec, plan.count_node, &count_params));

    cudaKernelNodeParams prefix_params{};
    void* prefix_args[] = {&counts_const_arg, &block_offsets_arg, &expert_offsets_arg, &n_blocks_arg};
    prefix_params.func = reinterpret_cast<void*>(prefix_counts_kernel);
    prefix_params.gridDim = dim3(kExperts);
    prefix_params.blockDim = dim3(kThreads);
    prefix_params.sharedMemBytes = 0;
    prefix_params.kernelParams = prefix_args;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(plan.exec, plan.prefix_node, &prefix_params));

    cudaKernelNodeParams expert_params{};
    void* expert_args[] = {&expert_offsets_arg};
    expert_params.func = reinterpret_cast<void*>(expert_offsets_kernel);
    expert_params.gridDim = dim3(1);
    expert_params.blockDim = dim3(kThreads);
    expert_params.sharedMemBytes = 0;
    expert_params.kernelParams = expert_args;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(plan.exec, plan.expert_node, &expert_params));

    cudaKernelNodeParams scatter_params{};
    void* scatter_args[] = {
        &topk_arg,
        &sorted_arg,
        &block_offsets_const_arg,
        &expert_offsets_const_arg,
        &n_arg,
    };
    scatter_params.func = reinterpret_cast<void*>(scatter_tiles_kernel);
    scatter_params.gridDim = dim3(plan.n_blocks);
    scatter_params.blockDim = dim3(kThreads);
    scatter_params.sharedMemBytes = 0;
    scatter_params.kernelParams = scatter_args;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(plan.exec, plan.scatter_node, &scatter_params));

    CUDA_CHECK(cudaGraphLaunch(plan.exec, stream));
}

}  // namespace

void run(
    torch::Tensor topk_idx,
    torch::Tensor sorted_token_indices,
    torch::Tensor expert_offsets) {
    const at::cuda::CUDAGuard guard(topk_idx.device());
    const int n = static_cast<int>(topk_idx.numel());
    const int n_blocks = (n + kTile - 1) / kTile;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto& counts = scratch_tensor(0, topk_idx.device());
    auto& block_offsets = scratch_tensor(1, topk_idx.device());
    GraphPlan& plan = graph_plan(topk_idx.device().index(), n, n_blocks);

    launch_graph(
        plan,
        topk_idx.data_ptr<int>(),
        counts.data_ptr<int>(),
        block_offsets.data_ptr<int>(),
        sorted_token_indices.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "SOL58 MoE expert stable counting sort (CUDA graph DPS)");
}
