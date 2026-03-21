#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>

#include "func_interface.h"

// CUDA error checking macro
#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

// XXX
// Data structure to hold all buffers
struct Buffers {
  // Host buffers
  float* h_input;
  float* h_output;

  // Device buffers
  float* d_input;
  float* d_output;

  Params params;

  // Constructor: allocate all buffers
  Buffers(const Params& p) : params(p) {
    size_t input_bytes = params.input_size() * sizeof(float);
    size_t output_bytes = params.output_size() * sizeof(float);

    // Allocate host memory
    h_input = new float[params.input_size()];
    h_output = new float[params.output_size()];

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    // Initialize with random data
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < params.input_size(); i++) {
      h_input[i] = dis(gen);
    }
  }

  // Destructor: free all buffers
  ~Buffers() {
    delete[] h_input;
    delete[] h_output;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
  }

  // Prevent copying (expensive)
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;
};

void compare_outputs(const float* output1, const float* output2, size_t size,
                     float tol = 1e-3f) {
  float max_diff = 0.0f;
  float rel_error = 0.0f;
  for (int i = 0; i < size; i++) {
    float diff = std::abs(output1[i] - output2[i]);
    max_diff = std::max(max_diff, diff);

    // If value is very small, avoid division by zero
    rel_error =
        std::max(rel_error, diff / std::max(std::abs(output2[i]), 1e-5f));
  }

  printf("  Max absolute difference: %.2e\n", max_diff);
  printf("  Max relative error: %.2e\n", rel_error);
}

// Timing result with breakdown
struct TimingResult {
  float h2d_time;     // Host to Device transfer
  float kernel_time;  // Kernel execution
  float d2h_time;     // Device to Host transfer
  float total_time;   // Total end-to-end time

  void print(const char* name, float baseline_time = 0) const {
    float speedup = (baseline_time > 0) ? baseline_time / total_time : 1.0f;

    printf("%-25s: %8.3f ms", name, total_time);
    if (baseline_time > 0) {
      printf("  (%6.2fx speedup)", speedup);
    }
    printf("\n");

    printf("  └─ H2D: %.3f ms, Kernel: %.3f ms, D2H: %.3f ms\n", h2d_time,
           kernel_time, d2h_time);
  }
};

// Time a CPU function
template <typename Func>
TimingResult time_cpu_function(Func XXXfunc, Buffers& buffers,
                               int warmup_iters = 2, int timing_iters = 10) {
  TimingResult result = {0, 0, 0, 0};

  // Warmup
  for (int i = 0; i < warmup_iters; i++) {
    XXXfunc(buffers.h_input, buffers.h_output, buffers.params);
  }

  // Timed runs
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < timing_iters; i++) {
    XXXfunc(buffers.h_input, buffers.h_output, buffers.params);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration = end - start;

  result.total_time = duration.count() / timing_iters;
  result.kernel_time = result.total_time;  // CPU has no transfer overhead

  return result;
}

// Time a GPU function with full memory transfer
template <typename Func>
TimingResult time_gpu_function(Func XXXfunc, Buffers& buffers, int warmup_iters,
                               int timing_iters) {
  TimingResult result = {0, 0, 0, 0};

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Warmup
  for (int i = 0; i < warmup_iters; i++) {
    // XXX
    CUDA_CHECK(cudaMemcpy(buffers.d_input, buffers.h_input,
                          buffers.params.input_size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    XXXfunc(buffers.d_input, buffers.d_output, buffers.params);
    CUDA_CHECK(cudaMemcpy(buffers.h_output, buffers.d_output,
                          buffers.params.output_size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed runs
  for (int i = 0; i < timing_iters; i++) {
    printf("Iteration %d/%d\r", i + 1, timing_iters);
    float h2d, kernel, d2h;

    // Time H2D transfer
    CUDA_CHECK(cudaEventRecord(start));

    // XXX
    CUDA_CHECK(cudaMemcpy(buffers.d_input, buffers.h_input,
                          buffers.params.input_size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&h2d, start, stop));

    // Time kernel
    CUDA_CHECK(cudaEventRecord(start));

    // XXX
    XXXfunc(buffers.d_input, buffers.d_output, buffers.params);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel, start, stop));

    // Time D2H transfer
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(buffers.h_output, buffers.d_output,
                          buffers.params.output_size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&d2h, start, stop));

    result.h2d_time += h2d;
    result.kernel_time += kernel;
    result.d2h_time += d2h;
  }

  // Average over iterations
  result.h2d_time /= timing_iters;
  result.kernel_time /= timing_iters;
  result.d2h_time /= timing_iters;
  result.total_time = result.h2d_time + result.kernel_time + result.d2h_time;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return result;
}

// Calculate GFLOPS for function
inline float calculate_gflops(const Params& params, float time_ms) {
  // XXX
  // Calculate total operations
  long long ops = 0.0f;
  return (ops / 1e9) / (time_ms / 1e3);  // GFLOPS
}

#endif