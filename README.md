# CUDA Reduction Optimization Practice

This project demonstrates progressive optimization of CUDA parallel reduction kernels, comparing the performance and design tradeoffs of each approach. Each implementation stage builds understanding of why certain optimizations move the needle and why others don't — reduction being a canonical memory-bound problem.

## Project Overview

The project implements parallel sum reduction through different optimization stages, comparing performance metrics at each step. All kernel implementations use consistent parameters and inputs for fair comparison.

### Input Specifications
- **Input tensor**: 1D array of `float` values
- **Output**: Single scalar sum
- **Memory management**: CUDA device allocations with host verification against CPU reference
- **Timing measurement**: CUDA events record kernel execution time across implementations

## Implementation Stages

### 1. CPU Version
Baseline serialized implementation for correctness verification. All GPU kernels are benchmarked against this reference. Not competitive in throughput but useful for confirming numerical accuracy of each GPU stage.

### 2. Naive GPU
First GPU implementation. Each block reduces a portion of the input via a shared memory tree and atomically adds the block result to the output. Adjacent threads load adjacent elements to avoid coalescing issues on the load side.

Note: Despite the coalescing fix on loads, this is the baseline all other implementations are compared against. Reduction is a memory-bound problem, so the coalescing fix helps but the kernel is still bottlenecked by global memory bandwidth.

### 3. WarpReduce GPU
Introduces a stride-based load pattern and warp-level shuffle reduction for the final 32→1 step. Each thread accumulates multiple elements before entering the shared memory tree, reducing the number of elements that need to go through shared memory.

Note: Despite the theoretical improvements — fewer shared memory steps, warp shuffle intrinsics — this implementation performed about the same as naive on our test sizes. Both kernels are bottlenecked by global memory bandwidth. The synchronization and shuffle savings are a tiny fraction of total kernel time compared to the cost of loading N floats from DRAM. The warp shuffle optimization matters more inside larger kernels (e.g. softmax, layernorm) where the reduction is over a small segment and not the bottleneck itself.

Note 2: A subtle correctness issue exists if loading stale smem values into threads outside the first warp before the shuffle step. Only thread 0 issues the atomicAdd so it does not affect correctness, but it is a latent bug worth fixing explicitly.

### 4. Modern GPU
A ground-up rewrite targeting the actual bottleneck: memory bandwidth and atomicAdd contention. Three changes attack this directly.

First, `float4` vectorized loads replace scalar loads. Each memory transaction now moves 128 bits instead of 32, reducing the number of L2 cache requests by 4× for the same data volume.

Second, a grid-stride loop with an SM-aware grid cap replaces the fixed one-block-per-chunk mapping. The grid is capped at `SM_count × 2` blocks regardless of N. Each block loops over its share of the input rather than launching thousands of blocks that each do a small amount of work. This directly reduces atomicAdd contention since far fewer blocks are racing to update the output.

Third, the shared memory tree is replaced entirely with warp shuffle reductions. After the grid-stride accumulation, each warp reduces its partial sum via `__shfl_down_sync` entirely in registers. Only 8 floats (one per warp in a 256-thread block) are ever written to shared memory, requiring a single `__syncthreads()` for the inter-warp handoff. Thread 0 issues one atomicAdd per block.

Note: The `float4` loads and the grid-stride cap are what actually move the needle. The shuffle-only reduction is cleaner than the smem tree but the savings are small relative to the bandwidth gains.

Note 2: `__shfl_down_sync` is preferred over `__shfl_xor_sync` for a plain reduction. The butterfly pattern of xor-sync leaves every lane with the sum, which wastes work when only lane 0 needs the result.

### 5. cuBLAS / Thrust Reference (future work)
Comparison against a production-grade implementation. This will serve as the performance ceiling to understand how much headroom remains above the modern kernel.

## Build Instructions

### Prerequisites
- CUDA Toolkit (with nvcc compiler)
- g++ compiler
- GNU Make

### Building the Project

```bash
# Build all targets
make

# Run the benchmark
make run
```

### Available Targets

```bash
make          # Build the project
make run      # Build and run the benchmark
make clean    # Remove build artifacts
```

## Performance Analysis Workflow

1. Run CPU reference to establish a correctness baseline
2. Run naive GPU and record execution time
3. Implement the next optimization stage, compare timing against previous implementations
4. Reason about why the numbers changed or didn't — is the bottleneck bandwidth, compute, or contention?
5. Iterate

## Key Learning Points

- Reduction is a memory bandwidth-bound problem. Optimizing synchronization and compute when the kernel is sitting idle waiting on DRAM yields no measurable improvement.
- Warp shuffle intrinsics are most impactful inside larger kernels where the reduction is a small segment, not the bottleneck.
- The number of atomicAdd calls per block matters. Naive warp-level implementations can accidentally issue one atomic per warp (8×) rather than one per block if control flow is not careful.
- `float4` vectorized loads are the single highest-leverage change for bandwidth-bound kernels.
- Grid-stride loops with SM-aware grid sizing reduce atomic contention and are portable across GPU generations without hand-tuning.
- `__shfl_down_sync` vs `__shfl_xor_sync`: use down-sync for reductions where only lane 0 needs the result; xor-sync (butterfly) wastes work in that case.

## Future Work

- Vectorized loads with tail handling for arbitrary N
- Two-pass reduction to eliminate atomicAdd entirely (write per-block results to a temp buffer, reduce in a second kernel)
- Performance analysis across different input sizes to find the crossover point where bandwidth saturates
- Register tiling to increase arithmetic intensity
- Comparison against Thrust and CUB `DeviceReduce` as production-grade ceilings