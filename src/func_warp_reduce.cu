#include "func_interface.h"

#define BLOCKSIZE 256
#define STRIDE 8
#define WARPSIZE 32

namespace warpReduce {

// To solve coalescing issues, adjacent threads will load adjacent elements
__global__ void reduction_kernel(const float* input, float* output,
                                 const size_t N) {
  __shared__ float smem[BLOCKSIZE];
  int tx = threadIdx.x;

  // Step 1
  // Reducing number of elements in a block by STRIDE
  // BLOCKSIZE * STRIDE -> BLOCKSIZE
  float sum = 0;
  int block_offset = blockIdx.x * BLOCKSIZE * STRIDE;
#pragma unroll
  for (int i = 0; i < STRIDE; i++) {
    int idx = block_offset + i * BLOCKSIZE + tx;
    if (idx < N) {
      sum += input[idx];
    }
  }
  smem[tx] = sum;
  __syncthreads();

  // Step 2
  // Classic reduction down to 32
  // BLOCKSIZE -> WARPSIZE
#pragma unroll
  for (int stride = BLOCKSIZE / 2; stride >= WARPSIZE; stride >>= 1) {
    if (tx < stride) {
      smem[tx] += smem[tx + stride];
    }
    __syncthreads();
  }

  // Step 3
  // Warp level reduction
  // WARPSIZE -> 1
  // Atomic add to the final output`
  if (tx < WARPSIZE) {
    sum = smem[tx];
#pragma unroll
    for (int offset = WARPSIZE / 2; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
  }
  if (tx == 0) atomicAdd(output, sum);
}

/*
@brief
*/
void reduction(const float* input, float* output, const Params& params) {
  size_t N = params.input_size();

  dim3 dimblock(BLOCKSIZE);
  dim3 dimgrid((N + (BLOCKSIZE * STRIDE) - 1) / (BLOCKSIZE * STRIDE));

  reduction_kernel<<<dimgrid, dimblock>>>(input, output, N);
  cudaDeviceSynchronize();
}

}  // namespace warpReduce