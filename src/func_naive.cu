#include <stdio.h>

#include <algorithm>

#include "func_interface.h"

namespace naive {

// To solve coalescing issues, adjacent threads will load adjacent elements
__global__ void func_kernel() {}

void XXXfunc(const float* input, float* output, const Params& params) {}

}  // namespace naive