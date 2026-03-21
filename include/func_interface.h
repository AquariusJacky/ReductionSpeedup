#ifndef CONV_INTERFACE_H
#define CONV_INTERFACE_H

#include <cstddef>

// Parameters
struct Params {
  // Parameters for algorithm
  size_t num = 1 << 24;  // Default to 16M elements

  size_t input_size() const { return num; }
  size_t output_size() const { return 1; }
};

// Standard interface that ALL implementations must follow
// This makes benchmarking fair and consistent
void reduction(const float* input, float* output, const Params& params);

#endif