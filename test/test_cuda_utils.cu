#include "test_cuda_utils.h"

#include <cuda_runtime.h>

#include "gpu/kernel/primitives.h"

namespace gccl {
namespace {

extern "C" __global__ void Copy128bGlobal(CopyArgs args) {
  int tid = threadIdx.x;
  int n_threads = blockDim.x;
  args.tid = tid;
  args.n_threads = n_threads;
  Copy128b(&args);
}

}  // namespace
}  // namespace gccl