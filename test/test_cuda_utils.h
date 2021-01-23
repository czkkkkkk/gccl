#pragma once

#include <cuda_runtime.h>

#include "gpu/kernel/primitives.h"

namespace gccl {
namespace {

// FIXME Will have problem without extern C
extern "C" __global__ void Copy128bGlobal(CopyArgs args);

template <typename T>
std::vector<T> CopyCudaPtrToVec(T *dev_ptr, int size) {
  std::vector<T> ret(size);
  cudaMemcpy(ret.data(), dev_ptr, size * sizeof(T), cudaMemcpyDefault);
  return ret;
}

}  // namespace
}  // namespace gccl