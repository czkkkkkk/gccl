#pragma once

#include <cuda_runtime.h>

#include "core.h"

namespace gccl {

__global__ void GraphGreedyAllgatherKernel(CollectiveArgs args);
__global__ void GraphGreedyAllgatherBackwardKernel(CollectiveArgs args);

}  // namespace gccl