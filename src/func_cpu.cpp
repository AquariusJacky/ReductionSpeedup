#include "func_interface.h"

namespace cpu {

void reduction(const float* input, float* output, const Params& params) {
  output[0] = 0.0f;
  for (size_t i = 0; i < params.input_size(); i++) {
    output[0] += input[i];
  }
}

}  // namespace cpu