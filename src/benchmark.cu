#include "benchmark_utils.h"
#include "func_interface.h"

// Different implementation namespaces
namespace cpu {
void reduction(const float*, float*, const Params&);
}
namespace naive {
void reduction(const float*, float*, const Params&);
}
namespace warpReduce {
void reduction(const float*, float*, const Params&);
}
namespace modernReduce {
void reduction(const float*, float*, const Params&);
}

int main(int argc, char** argv) {
  // Define problem size
  Params params;

  // Report mode or comparison mode
  bool report_mode = false;
  bool naive_gpu = true;
  bool warp_reduce = true;
  bool modern_reduce = true;

  int warmup_iters, timing_iters;
  if (report_mode) {
    warmup_iters = 2;
    timing_iters = 5;
  } else {
    warmup_iters = 5;
    timing_iters = 20;
  }

  // Allocate all buffers once (via RAII)
  Buffers buffers(params);

  // Print configuration
  printf("Reduction: %zu elements\n", params.input_size());
  printf("Output Total FLOPs: %.2f GFLOP\n\n",
         calculate_gflops(params, 1000.0f));  // GFLOPS at 1 second

  TimingResult cpu_result, naive_result, warp_reduce_result,
      modern_reduce_result;
  float *cpu_output, *naive_output, *warp_reduce_output, *modern_reduce_output;

  // Benchmark CPU
  cpu_result = time_cpu_reduction(cpu::reduction, buffers);
  cpu_output = new float[params.output_size()];
  memcpy(cpu_output, buffers.h_output, params.output_size() * sizeof(float));

  if (naive_gpu) {  // Unoptimized GPU implementations
    naive_result = time_gpu_reduction(naive::reduction, buffers, warmup_iters,
                                      timing_iters);
    naive_output = new float[params.output_size()];
    memcpy(naive_output, buffers.h_output,
           params.output_size() * sizeof(float));

    printf("Comparing naive output to CPU reference...\n");
    compare_outputs(cpu_output, naive_output, params.output_size());
  }

  if (warp_reduce) {
    warp_reduce_result = time_gpu_reduction(warpReduce::reduction, buffers,
                                            warmup_iters, timing_iters);
    warp_reduce_output = new float[params.output_size()];
    memcpy(warp_reduce_output, buffers.h_output,
           params.output_size() * sizeof(float));

    printf("Comparing warp reduce output to CPU reference...\n");
    compare_outputs(cpu_output, warp_reduce_output, params.output_size());
  }

  if (modern_reduce) {
    modern_reduce_result = time_gpu_reduction(modernReduce::reduction, buffers,
                                              warmup_iters, timing_iters);
    modern_reduce_output = new float[params.output_size()];
    memcpy(modern_reduce_output, buffers.h_output,
           params.output_size() * sizeof(float));

    printf("Comparing modern reduce output to CPU reference...\n");
    compare_outputs(cpu_output, modern_reduce_output, params.output_size());
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  printf("\nPerformance:\n");

  // Print results
  cpu_result.print("Baseline");
  if (naive_gpu) naive_result.print("Naive", cpu_result.total_time);
  if (warp_reduce)
    warp_reduce_result.print("Warp Reduce", cpu_result.total_time);
  if (modern_reduce) {
    modern_reduce_result.print("Modern Reduce", cpu_result.total_time);
  }

  // Print GFLOPS
  printf("  CPU:    %.2f GFLOPS\n",
         calculate_gflops(params, cpu_result.total_time));
  if (naive_gpu)
    printf("  Naive:  %.2f GFLOPS\n",
           calculate_gflops(params, naive_result.kernel_time));

  if (warp_reduce)
    printf("  Warp Reduce GFLOPS: %.2f GFLOPS\n",
           calculate_gflops(params, warp_reduce_result.kernel_time));
  if (modern_reduce)
    printf("  Modern Reduce GFLOPS: %.2f GFLOPS\n",
           calculate_gflops(params, modern_reduce_result.kernel_time));

  delete[] cpu_output;
  if (naive_output) delete[] naive_output;
  if (warp_reduce_output) delete[] warp_reduce_output;
  if (modern_reduce_output) delete[] modern_reduce_output;

  return 0;
}