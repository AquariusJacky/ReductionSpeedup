#include "func_interface.h"

#define BLOCKSIZE 256

namespace naive {

// To solve coalescing issues, adjacent threads will load adjacent elements
__global__ void reduction_kernel(const float* input, float* output,
                                 const size_t N) {
  __shared__ float smem[BLOCKSIZE * 2];
  size_t tx = threadIdx.x;
  size_t idx = blockDim.x * blockIdx.x * 2 + tx;

  if (idx < N)
    smem[tx] = input[idx];
  else
    smem[tx] = 0.0f;  // Handle out-of-bounds threads
  if (idx + blockDim.x < N)
    smem[tx + blockDim.x] = input[idx + blockDim.x];
  else
    smem[tx + blockDim.x] = 0.0f;  // Handle out-of-bounds threads
  __syncthreads();

  for (size_t stride = blockDim.x; stride > 0; stride >>= 1) {
    if (tx < stride) {
      smem[tx] += smem[tx + stride];
    }
    __syncthreads();
  }

  if (tx == 0) atomicAdd(output, smem[0]);
}

/*
@brief Naive reduction implementation that performs a simple parallel reduction
        without any optimizations. Each block reduces a portion of the input via
        shared memory and atomically adds the block's result to the output.
*/
void reduction(const float* input, float* output, const Params& params) {
  size_t N = params.input_size();

  dim3 dimblock(BLOCKSIZE);
  dim3 dimgrid((N + (dimblock.x * 2) - 1) / (dimblock.x * 2));

  reduction_kernel<<<dimgrid, dimblock>>>(input, output, N);
  cudaDeviceSynchronize();
}

}  // namespace naive