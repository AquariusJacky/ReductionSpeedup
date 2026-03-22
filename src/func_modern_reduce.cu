#include "func_interface.h"

#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 16

namespace modernReduce {

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

__global__ void reduction_kernel(const float* __restrict__ input,
                                 float* __restrict__ output, const size_t N) {
  __shared__ float warp_sums[BLOCK_SIZE / 32];  // One slot per warp

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
  const int grid_stride = gridDim.x * BLOCK_SIZE;

  // Step 1: Each thread accumulates ITEMS_PER_THREAD elements
  // via float4 vectorized loads (128-bit per transaction)
  float thread_sum = 0.0f;

  // Grid-stride loop — each iteration jumps a full grid width
  int base = global_tid * (ITEMS_PER_THREAD / 4);
  int jump = grid_stride * (ITEMS_PER_THREAD / 4);

  while (base * 4 < (int)N) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD / 4; i++) {
      int idx = base + i;
      if ((idx + 1) * 4 <= (int)N) {
        float4 v = reinterpret_cast<const float4*>(input)[idx];
        thread_sum += v.x + v.y + v.z + v.w;
      } else {
        // Scalar tail: handles N not divisible by 4
        for (int j = idx * 4; j < min((int)N, idx * 4 + 4); j++)
          thread_sum += input[j];
      }
    }
    base += jump;
  }

  // Step 2: Warp-level reduce via shuffle — no __syncthreads() needed
  thread_sum = warp_reduce_sum(thread_sum);

  // Step 3: Lane 0 of each warp writes its result to shared memory
  if (lane_id == 0) warp_sums[warp_id] = thread_sum;
  __syncthreads();

  // Step 4: First warp reduces the warp_sums array (BLOCK_SIZE/32 = 8 values)
  // Load only if this lane has a valid warp result, else 0
  float val = (tid < BLOCK_SIZE / 32) ? warp_sums[tid] : 0.0f;
  if (warp_id == 0) val = warp_reduce_sum(val);

  // Step 5: Single atomic per block (not per warp)
  if (tid == 0) atomicAdd(output, val);
}

/*
@brief This method performs reduction on the GPU using a more optimized
warp-level approach that minimizes shared memory usage and synchronization. Each
thread processes multiple elements using vectorized loads, then performs a
warp-level reduction using shuffle instructions, and finally only one atomic add
per block is performed to update the output.
*/
void reduction(const float* input, float* output, const Params& params) {
  size_t N = params.input_size();

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  int max_blocks = prop.multiProcessorCount * 2;
  int needed = (N + (size_t)(BLOCK_SIZE * ITEMS_PER_THREAD) - 1) /
               (BLOCK_SIZE * ITEMS_PER_THREAD);
  int grid_size = min(needed, max_blocks);

  reduction_kernel<<<grid_size, BLOCK_SIZE>>>(input, output, N);
}

}  // namespace modernReduce