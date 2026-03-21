#ifndef CONV_INTERFACE_H
#define CONV_INTERFACE_H

#include <cstddef>

// Parameters
struct Params {
  // XXX
  // Parameters for algorithm

  size_t input_size() const {
    // XXX
    return 0;
  }
  size_t output_size() const {
    // XXX
    return 0;
  }
};

// Standard interface that ALL implementations must follow
// This makes benchmarking fair and consistent
void XXXfunc(const float* input, float* output, const Params& params);

#endif