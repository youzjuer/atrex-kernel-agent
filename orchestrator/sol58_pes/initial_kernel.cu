#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <unordered_map>

namespace {

namespace cg = cooperative_groups;

constexpr int kExperts = 256;
constexpr int kTile = 256;
constexpr int kTile512 = 512;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kLogicalWarps512 = 16;
constexpr int kExpertWords = kExperts / 32;
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
        if (lane >= offset) v += add;
    }
    return v;
}

__device__ __forceinline__ int block_inclusive_scan_256(int v, int tid, int* warp_totals) {
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int scan = warp_inclusive_scan(v, lane);
    if (lane == 31) warp_totals[warp] = scan;
    __syncthreads();
    if (warp == 0) {
        int wt = lane < kWarps ? warp_totals[lane] : 0;
        int wp = warp_inclusive_scan(wt, lane);
        if (lane < kWarps) warp_totals[lane] = wp - wt;
    }
    __syncthreads();
    return scan + warp_totals[warp];
}

__device__ __forceinline__ int group_scan_64(int v, int tid, int* warp_totals) {
    const int group = tid >> 6;
    const int local = tid & 63;
    const int lane = tid & 31;
    const int half = local >> 5;
    int scan = warp_inclusive_scan(v, lane);
    if (lane == 31) warp_totals[group * 2 + half] = scan;
    __syncthreads();
    int base = (half == 1) ? warp_totals[group * 2] : 0;
    return scan + base;
}

__device__ __forceinline__ int group_scan_96(int v, int tid, int* tmp) {
    const int lane = tid & 31;
    const int group = tid / 96;
    const int local = tid - group * 96;
    const bool valid = (group < 2) && (local < 96);
    const int warp_in_group = local >> 5;
    int scan = warp_inclusive_scan(valid ? v : 0, lane);
    if (valid && lane == 31) tmp[group * 3 + warp_in_group] = scan;
    __syncthreads();
    if (valid && warp_in_group == 0) {
        int wt = (lane < 3) ? tmp[group * 3 + lane] : 0;
        int wp = warp_inclusive_scan(wt, lane);
        if (lane < 3) tmp[6 + group * 3 + lane] = wp - wt;
    }
    __syncthreads();
    int base = valid ? tmp[6 + group * 3 + warp_in_group] : 0;
    return valid ? (scan + base) : 0;
}

__device__ __forceinline__ int group_scan_128(int v, int tid, int* warp_totals) {
    const int group = tid >> 7;
    const int local = tid & 127;
    const int lane = tid & 31;
    const int warp_in_group = local >> 5;
    int scan = warp_inclusive_scan(v, lane);
    if (lane == 31) warp_totals[group * 4 + warp_in_group] = scan;
    __syncthreads();
    if (warp_in_group == 0) {
        int wt = (lane < 4) ? warp_totals[group * 4 + lane] : 0;
        int wp = warp_inclusive_scan(wt, lane);
        if (lane < 4) warp_totals[group * 4 + lane] = wp - wt;
    }
    __syncthreads();
    return scan + warp_totals[group * 4 + warp_in_group];
}

template<int MODE>
__global__ __launch_bounds__(kThreads, 2)
void coop_sort256_kernel(
    const int* __restrict__ topk,
    int* __restrict__ sorted_token_indices,
    int* __restrict__ counts,
    int* __restrict__ block_offsets,
    int* __restrict__ expert_offsets,
    int n) {
    cg::grid_group grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_blocks = (n + kTile - 1) / kTile;
    __shared__ int hist[kExperts];
    __shared__ int scan_tmp[12];
    __shared__ int warp_counts[kWarps * kExperts];
    __shared__ unsigned int warp_mask[kWarps * kExpertWords];
    const int idx = bid * kTile + tid;
    const bool tile_active = bid < n_blocks;
    const bool active = tile_active && (idx < n);
    const int expert = active ? topk[idx] : 0;
    unsigned int same_warp = 0u;
    if (tile_active) {
        hist[tid] = 0;
        if (tid < kWarps * kExpertWords) warp_mask[tid] = 0u;
        __syncthreads();
        unsigned int active_mask = __ballot_sync(0xffffffffu, active);
        if (active_mask != 0u) {
            same_warp = __match_any_sync(active_mask, expert);
            int leader = __ffs(same_warp) - 1;
            if (active && lane == leader) {
                const int cnt = __popc(same_warp);
                atomicAdd(&hist[expert], cnt);
                warp_counts[warp * kExperts + expert] = cnt;
                atomicOr(&warp_mask[warp * kExpertWords + (expert >> 5)], 1u << (expert & 31));
            }
        }
        __syncthreads();
        counts[bid * kExperts + tid] = hist[tid];
    } else {
        counts[bid * kExperts + tid] = 0;
    }
    grid.sync();
    if constexpr (MODE == 64) {
        const int group = tid >> 6;
        const int local = tid & 63;
        const int e = bid * 4 + group;
        int c = (local < n_blocks) ? counts[local * kExperts + e] : 0;
        int scan = group_scan_64(c, tid, scan_tmp);
        if (local < n_blocks) block_offsets[local * kExperts + e] = scan - c;
        if (local == 63) expert_offsets[e + 1] = scan;
    } else if constexpr (MODE == 96) {
        const bool prefix_valid = tid < 192;
        const int group = tid / 96;
        const int local = tid - group * 96;
        const int e = bid * 2 + group;
        int c = (prefix_valid && local < n_blocks) ? counts[local * kExperts + e] : 0;
        int scan = group_scan_96(c, tid, scan_tmp);
        if (prefix_valid && local < n_blocks) block_offsets[local * kExperts + e] = scan - c;
        if (prefix_valid && local == 95) expert_offsets[e + 1] = scan;
    } else if constexpr (MODE == 128) {
        const int group = tid >> 7;
        const int local = tid & 127;
        const int e = bid * 2 + group;
        int c = (local < n_blocks) ? counts[local * kExperts + e] : 0;
        int scan = group_scan_128(c, tid, scan_tmp);
        if (local < n_blocks) block_offsets[local * kExperts + e] = scan - c;
        if (local == 127) expert_offsets[e + 1] = scan;
    } else {
        const int e = bid;
        int c = counts[tid * kExperts + e];
        int scan = block_inclusive_scan_256(c, tid, scan_tmp);
        block_offsets[tid * kExperts + e] = scan - c;
        if (tid == kThreads - 1) expert_offsets[e + 1] = scan;
    }
    grid.sync();
    if (bid == 0) {
        int total = expert_offsets[tid + 1];
        int scan = block_inclusive_scan_256(total, tid, scan_tmp);
        if (tid == 0) expert_offsets[0] = 0;
        expert_offsets[tid + 1] = scan;
    }
    grid.sync();
    if (active) {
        const unsigned int lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
        const int rank_in_warp = __popc(same_warp & lane_mask);
        int prior_warps = 0;
        const int expert_word = expert >> 5;
        const unsigned int expert_bit = 1u << (expert & 31);
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            if (w < warp) {
                const unsigned int m = warp_mask[w * kExpertWords + expert_word];
                if (m & expert_bit) prior_warps += warp_counts[w * kExperts + expert];
            }
        }
        int out = expert_offsets[expert] + block_offsets[bid * kExperts + expert] + prior_warps + rank_in_warp;
        sorted_token_indices[out] = idx;
    }
}

template<int MODE>
__global__ __launch_bounds__(kThreads, 2)
void coop_sort512_kernel(
    const int* __restrict__ topk,
    int* __restrict__ sorted_token_indices,
    int* __restrict__ counts,
    int* __restrict__ block_offsets,
    int* __restrict__ expert_offsets,
    int n) {
    cg::grid_group grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_blocks = (n + kTile512 - 1) / kTile512;
    __shared__ int hist[kExperts];
    __shared__ int scan_tmp[12];
    __shared__ int warp_counts[kLogicalWarps512 * kExperts];
    __shared__ unsigned int warp_mask[kLogicalWarps512 * kExpertWords];

    const bool tile_active = bid < n_blocks;
    const int idx0 = bid * kTile512 + tid;
    const int idx1 = idx0 + kThreads;
    const bool active0 = tile_active && (idx0 < n);
    const bool active1 = tile_active && (idx1 < n);
    const int expert0 = active0 ? topk[idx0] : 0;
    const int expert1 = active1 ? topk[idx1] : 0;
    unsigned int same0 = 0u;
    unsigned int same1 = 0u;

    hist[tid] = 0;
    if (tid < kLogicalWarps512 * kExpertWords) warp_mask[tid] = 0u;
    __syncthreads();

    unsigned int mask0 = __ballot_sync(0xffffffffu, active0);
    if (mask0 != 0u) {
        same0 = __match_any_sync(mask0, expert0);
        const int leader = __ffs(same0) - 1;
        if (active0 && lane == leader) {
            const int cnt = __popc(same0);
            atomicAdd(&hist[expert0], cnt);
            warp_counts[warp * kExperts + expert0] = cnt;
            atomicOr(&warp_mask[warp * kExpertWords + (expert0 >> 5)], 1u << (expert0 & 31));
        }
    }
    unsigned int mask1 = __ballot_sync(0xffffffffu, active1);
    if (mask1 != 0u) {
        same1 = __match_any_sync(mask1, expert1);
        const int leader = __ffs(same1) - 1;
        if (active1 && lane == leader) {
            const int cnt = __popc(same1);
            const int lw = warp + kWarps;
            atomicAdd(&hist[expert1], cnt);
            warp_counts[lw * kExperts + expert1] = cnt;
            atomicOr(&warp_mask[lw * kExpertWords + (expert1 >> 5)], 1u << (expert1 & 31));
        }
    }
    __syncthreads();
    counts[bid * kExperts + tid] = tile_active ? hist[tid] : 0;

    grid.sync();
    if constexpr (MODE == 64) {
        const int group = tid >> 6;
        const int local = tid & 63;
        const int e = bid * 4 + group;
        int c = (local < n_blocks) ? counts[local * kExperts + e] : 0;
        int scan = group_scan_64(c, tid, scan_tmp);
        if (local < n_blocks) block_offsets[local * kExperts + e] = scan - c;
        if (local == 63) expert_offsets[e + 1] = scan;
    } else {
        const int group = tid >> 7;
        const int local = tid & 127;
        const int e = bid * 2 + group;
        int c = (local < n_blocks) ? counts[local * kExperts + e] : 0;
        int scan = group_scan_128(c, tid, scan_tmp);
        if (local < n_blocks) block_offsets[local * kExperts + e] = scan - c;
        if (local == 127) expert_offsets[e + 1] = scan;
    }
    grid.sync();
    if (bid == 0) {
        int total = expert_offsets[tid + 1];
        int scan = block_inclusive_scan_256(total, tid, scan_tmp);
        if (tid == 0) expert_offsets[0] = 0;
        expert_offsets[tid + 1] = scan;
    }
    grid.sync();

    const unsigned int lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
    if (active0) {
        const int rank_in_warp = __popc(same0 & lane_mask);
        int prior = 0;
        const int expert_word = expert0 >> 5;
        const unsigned int expert_bit = 1u << (expert0 & 31);
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            if (w < warp) {
                const unsigned int m = warp_mask[w * kExpertWords + expert_word];
                if (m & expert_bit) prior += warp_counts[w * kExperts + expert0];
            }
        }
        const int out = expert_offsets[expert0] + block_offsets[bid * kExperts + expert0] + prior + rank_in_warp;
        sorted_token_indices[out] = idx0;
    }
    if (active1) {
        const int rank_in_warp = __popc(same1 & lane_mask);
        int prior = 0;
        const int expert_word = expert1 >> 5;
        const unsigned int expert_bit = 1u << (expert1 & 31);
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            const unsigned int m = warp_mask[w * kExpertWords + expert_word];
            if (m & expert_bit) prior += warp_counts[w * kExperts + expert1];
        }
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            if (w < warp) {
                const int lw = w + kWarps;
                const unsigned int m = warp_mask[lw * kExpertWords + expert_word];
                if (m & expert_bit) prior += warp_counts[lw * kExperts + expert1];
            }
        }
        const int out = expert_offsets[expert1] + block_offsets[bid * kExperts + expert1] + prior + rank_in_warp;
        sorted_token_indices[out] = idx1;
    }
}

__global__ __launch_bounds__(kThreads, 2)
void count_tiles_kernel(const int* __restrict__ topk, int* __restrict__ counts, int n) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * kTile + tid;
    const int lane = tid & 31;
    __shared__ int hist[kExperts];
    hist[tid] = 0;
    __syncthreads();
    bool active = idx < n;
    int expert = active ? topk[idx] : 0;
    unsigned int active_mask = __ballot_sync(0xffffffffu, active);
    if (active_mask != 0u) {
        unsigned int same = __match_any_sync(active_mask, expert);
        int leader = __ffs(same) - 1;
        if (active && lane == leader) atomicAdd(&hist[expert], __popc(same));
    }
    __syncthreads();
    counts[blockIdx.x * kExperts + tid] = hist[tid];
}

__global__ __launch_bounds__(kThreads, 2)
void prefix_counts_kernel(const int* __restrict__ counts, int* __restrict__ block_offsets, int* __restrict__ expert_offsets, int n_blocks) {
    const int expert = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int warp_totals[kWarps];
    int c = tid < n_blocks ? counts[tid * kExperts + expert] : 0;
    int scan = block_inclusive_scan_256(c, tid, warp_totals);
    if (tid < n_blocks) block_offsets[tid * kExperts + expert] = scan - c;
    if (tid == kThreads - 1) expert_offsets[expert + 1] = scan;
}

__global__ __launch_bounds__(kThreads, 1)
void expert_offsets_kernel(int* __restrict__ expert_offsets) {
    const int tid = threadIdx.x;
    __shared__ int warp_totals[kWarps];
    int total = expert_offsets[tid + 1];
    int scan = block_inclusive_scan_256(total, tid, warp_totals);
    if (tid == 0) expert_offsets[0] = 0;
    expert_offsets[tid + 1] = scan;
}

__global__ __launch_bounds__(kThreads, 2)
void scatter_tiles_kernel(const int* __restrict__ topk, int* __restrict__ sorted_token_indices, const int* __restrict__ block_offsets, const int* __restrict__ expert_offsets, int n) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * kTile + tid;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned int lane_mask = lane == 0 ? 0u : ((1u << lane) - 1u);
    __shared__ int warp_hist[kWarps * kExperts];
    for (int i = tid; i < kWarps * kExperts; i += kThreads) warp_hist[i] = 0;
    __syncthreads();
    bool active = idx < n;
    int expert = active ? topk[idx] : 0;
    unsigned int active_mask = __ballot_sync(0xffffffffu, active);
    unsigned int same_warp = 0u;
    if (active_mask != 0u) {
        same_warp = __match_any_sync(active_mask, expert);
        int leader = __ffs(same_warp) - 1;
        if (active && lane == leader) warp_hist[warp * kExperts + expert] = __popc(same_warp);
    }
    __syncthreads();
    const int rank_in_warp = active ? __popc(same_warp & lane_mask) : 0;
    int prior_warps = 0;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) if (w < warp && active) prior_warps += warp_hist[w * kExperts + expert];
    if (active) {
        int out = expert_offsets[expert] + block_offsets[blockIdx.x * kExperts + expert] + prior_warps + rank_in_warp;
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
    if (it != plans.end()) return it->second;
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
    count_params.kernelParams = count_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&plan.count_node, plan.graph, nullptr, 0, &count_params));
    dummy_counts_const = dummy_counts;
    cudaKernelNodeParams prefix_params{};
    void* prefix_args[] = {&dummy_counts_const, &dummy_block_offsets, &dummy_expert_offsets, &n_blocks_arg};
    prefix_params.func = reinterpret_cast<void*>(prefix_counts_kernel);
    prefix_params.gridDim = dim3(kExperts);
    prefix_params.blockDim = dim3(kThreads);
    prefix_params.kernelParams = prefix_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&plan.prefix_node, plan.graph, &plan.count_node, 1, &prefix_params));
    cudaKernelNodeParams expert_params{};
    void* expert_args[] = {&dummy_expert_offsets};
    expert_params.func = reinterpret_cast<void*>(expert_offsets_kernel);
    expert_params.gridDim = dim3(1);
    expert_params.blockDim = dim3(kThreads);
    expert_params.kernelParams = expert_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&plan.expert_node, plan.graph, &plan.prefix_node, 1, &expert_params));
    dummy_block_offsets_const = dummy_block_offsets;
    dummy_expert_offsets_const = dummy_expert_offsets;
    cudaKernelNodeParams scatter_params{};
    void* scatter_args[] = {&dummy_topk, &dummy_sorted, &dummy_block_offsets_const, &dummy_expert_offsets_const, &n_arg};
    scatter_params.func = reinterpret_cast<void*>(scatter_tiles_kernel);
    scatter_params.gridDim = dim3(n_blocks);
    scatter_params.blockDim = dim3(kThreads);
    scatter_params.kernelParams = scatter_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&plan.scatter_node, plan.graph, &plan.expert_node, 1, &scatter_params));
    CUDA_CHECK(cudaGraphInstantiate(&plan.exec, plan.graph, 0));
    auto inserted = plans.emplace(key, plan);
    return inserted.first->second;
}

void launch_graph(GraphPlan& plan, const int* topk, int* counts, int* block_offsets, int* sorted_token_indices, int* expert_offsets, cudaStream_t stream) {
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
    count_params.kernelParams = count_args;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(plan.exec, plan.count_node, &count_params));
    cudaKernelNodeParams prefix_params{};
    void* prefix_args[] = {&counts_const_arg, &block_offsets_arg, &expert_offsets_arg, &n_blocks_arg};
    prefix_params.func = reinterpret_cast<void*>(prefix_counts_kernel);
    prefix_params.gridDim = dim3(kExperts);
    prefix_params.blockDim = dim3(kThreads);
    prefix_params.kernelParams = prefix_args;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(plan.exec, plan.prefix_node, &prefix_params));
    cudaKernelNodeParams expert_params{};
    void* expert_args[] = {&expert_offsets_arg};
    expert_params.func = reinterpret_cast<void*>(expert_offsets_kernel);
    expert_params.gridDim = dim3(1);
    expert_params.blockDim = dim3(kThreads);
    expert_params.kernelParams = expert_args;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(plan.exec, plan.expert_node, &expert_params));
    cudaKernelNodeParams scatter_params{};
    void* scatter_args[] = {&topk_arg, &sorted_arg, &block_offsets_const_arg, &expert_offsets_const_arg, &n_arg};
    scatter_params.func = reinterpret_cast<void*>(scatter_tiles_kernel);
    scatter_params.gridDim = dim3(plan.n_blocks);
    scatter_params.blockDim = dim3(kThreads);
    scatter_params.kernelParams = scatter_args;
    CUDA_CHECK(cudaGraphExecKernelNodeSetParams(plan.exec, plan.scatter_node, &scatter_params));
    CUDA_CHECK(cudaGraphLaunch(plan.exec, stream));
}

template<int MODE, int GRID>
cudaError_t launch_coop256(const int* topk, int* sorted_token_indices, int* counts, int* block_offsets, int* expert_offsets, int n, cudaStream_t stream) {
    const int* topk_arg = topk;
    int* sorted_arg = sorted_token_indices;
    int* counts_arg = counts;
    int* block_offsets_arg = block_offsets;
    int* expert_offsets_arg = expert_offsets;
    int n_arg = n;
    void* args[] = {&topk_arg, &sorted_arg, &counts_arg, &block_offsets_arg, &expert_offsets_arg, &n_arg};
    return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(coop_sort256_kernel<MODE>), dim3(GRID), dim3(kThreads), args, 0, stream);
}

template<int MODE, int GRID>
cudaError_t launch_coop512(const int* topk, int* sorted_token_indices, int* counts, int* block_offsets, int* expert_offsets, int n, cudaStream_t stream) {
    const int* topk_arg = topk;
    int* sorted_arg = sorted_token_indices;
    int* counts_arg = counts;
    int* block_offsets_arg = block_offsets;
    int* expert_offsets_arg = expert_offsets;
    int n_arg = n;
    void* args[] = {&topk_arg, &sorted_arg, &counts_arg, &block_offsets_arg, &expert_offsets_arg, &n_arg};
    return cudaLaunchCooperativeKernel(reinterpret_cast<void*>(coop_sort512_kernel<MODE>), dim3(GRID), dim3(kThreads), args, 0, stream);
}

}  // namespace

void run(torch::Tensor topk_idx, torch::Tensor sorted_token_indices, torch::Tensor expert_offsets) {
    const at::cuda::CUDAGuard guard(topk_idx.device());
    const int n = static_cast<int>(topk_idx.numel());
    const int n_blocks = (n + kTile - 1) / kTile;
    const int n_blocks512 = (n + kTile512 - 1) / kTile512;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto& counts = scratch_tensor(0, topk_idx.device());
    auto& block_offsets = scratch_tensor(1, topk_idx.device());
    const int device_index = topk_idx.device().index();
    const int* topk = topk_idx.data_ptr<int>();
    int* sorted = sorted_token_indices.data_ptr<int>();
    int* counts_ptr = counts.data_ptr<int>();
    int* block_offsets_ptr = block_offsets.data_ptr<int>();
    int* expert_offsets_ptr = expert_offsets.data_ptr<int>();

    cudaError_t err = cudaSuccess;
    // Devised from Plan: remove the harmful first-half shared histogram/atomic path and keep the proven dispatch/fallback topology.
    if (n_blocks > 128 && n_blocks512 <= 128) {
        err = launch_coop512<128, 128>(topk, sorted, counts_ptr, block_offsets_ptr, expert_offsets_ptr, n, stream);
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    if (n_blocks <= 64) {
        err = launch_coop256<64, 64>(topk, sorted, counts_ptr, block_offsets_ptr, expert_offsets_ptr, n, stream);
        if (err == cudaSuccess) return;
        cudaGetLastError();
    } else if (n_blocks <= 72) {
        err = launch_coop256<96, 128>(topk, sorted, counts_ptr, block_offsets_ptr, expert_offsets_ptr, n, stream);
        if (err == cudaSuccess) return;
        cudaGetLastError();
    } else if (n_blocks <= 128) {
        err = launch_coop256<128, 128>(topk, sorted, counts_ptr, block_offsets_ptr, expert_offsets_ptr, n, stream);
        if (err == cudaSuccess) return;
        cudaGetLastError();
    } else if (n_blocks <= 256) {
        err = launch_coop256<256, 256>(topk, sorted, counts_ptr, block_offsets_ptr, expert_offsets_ptr, n, stream);
        if (err == cudaSuccess) return;
        cudaGetLastError();
    }

    GraphPlan& plan = graph_plan(device_index, n, n_blocks);
    launch_graph(plan, topk, counts_ptr, block_offsets_ptr, sorted, expert_offsets_ptr, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "SOL58 stable MoE counting sort no-extra-state 512 scatter plus parent fallback CUDA");
}