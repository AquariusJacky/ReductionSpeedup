#include "benchmark_utils.h"
#include "func_interface.h"

// XXX Different implementations
namespace cpu {
void XXXfunc(const float*, float*, const Params&);
}
namespace naive {
void XXXfunc(const float*, float*, const Params&);
}

int main(int argc, char** argv) {
  // Define problem size
  Params params;

  // Report mode or comparison mode
  bool report_mode = false;
  bool naive_gpu = true;

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

  // XXX
  // Print configuration
  printf("Function: ");
  printf("Output Total FLOPs: %.2f GFLOP\n\n",
         calculate_gflops(params, 1000.0f));  // GFLOPS at 1 second

  TimingResult cpu_result, naive_result;
  float *cpu_output, *naive_output;

  // Benchmark CPU
  cpu_result = time_cpu_function(cpu::XXXfunc, buffers);
  cpu_output = new float[params.output_size()];
  memcpy(cpu_output, buffers.h_output, params.output_size() * sizeof(float));

  if (naive_gpu) {  // Unoptimized GPU implementations
    naive_result =
        time_gpu_function(naive::XXXfunc, buffers, warmup_iters, timing_iters);
    naive_output = new float[params.output_size()];
    memcpy(naive_output, buffers.h_output,
           params.output_size() * sizeof(float));

    printf("Comparing naive output to CPU reference...\n");
    compare_outputs(cpu_output, naive_output, params.output_size());
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  printf("\nPerformance:\n");

  // Print results
  cpu_result.print("Baseline");
  if (naive_gpu) naive_result.print("Naive", cpu_result.total_time);

  // Print GFLOPS
  printf("  CPU:    %.2f GFLOPS\n",
         calculate_gflops(params, cpu_result.total_time));
  if (naive_gpu)
    printf("  Naive:  %.2f GFLOPS\n",
           calculate_gflops(params, naive_result.kernel_time));

  delete[] cpu_output;
  if (naive_output) delete[] naive_output;

  return 0;
}