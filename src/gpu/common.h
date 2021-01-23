#pragma once

#include <cuda_runtime.h>

#include <string>

#include "glog/logging.h"

#include "utils.h"

#define CUDACHECK(cmd)                                      \
  do {                                                      \
    cudaError_t e = cmd;                                    \
    if (e != cudaSuccess) {                                 \
      LOG(FATAL) << "Cuda error " << cudaGetErrorString(e); \
    }                                                       \
  } while (false);

#define CUDACHECKERR(e)                                     \
  do {                                                      \
    if (e != cudaSuccess) {                                 \
      LOG(FATAL) << "Cuda error " << cudaGetErrorString(e); \
    }                                                       \
  } while (false);

#ifdef GCCL_DEBUG
#define CUDA_DEBUG(cmd) \
  if (args->tid == 0) cmd
#else
#define CUDA_DEBUG(cmd)
#endif

namespace gccl {

template <typename T>
void GCCLCudaMalloc(T **ptr) {
  cudaMalloc(ptr, sizeof(T));
}

template <typename T>
void GCCLCudaMalloc(T **ptr, int size) {
  cudaMalloc(ptr, sizeof(T) * size);
  cudaMemset(*ptr, 0, sizeof(T) * size);
}

void GCCLSetCudaDevice(int dev_id);

template <typename T>
void GCCLMallocAndCopy(T **ret, const T *src, int size) {
  GCCLCudaMalloc(ret, size);
  cudaMemcpy(*ret, src, sizeof(T) * size, cudaMemcpyHostToDevice);
}

template <typename T>
void GCCLMallocAndCopy(T **ret, const std::vector<T> &src) {
  GCCLMallocAndCopy(ret, src.data(), src.size());
}

static inline void GCCLCudaHostAlloc(void **ptr, void **devPtr, size_t size) {
  CUDACHECK(cudaHostAlloc(ptr, size, cudaHostAllocMapped));
  memset(*ptr, 0, size);
  *devPtr = *ptr;
}

template <typename T>
std::string CudaVecToString(T *ptr, int size) {
  std::vector<T> vec(size);
  cudaMemcpy(vec.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
  return VecToString(vec);
}

template <typename T>
void GCCLCopyToCPU(T *dst, const T *src, int size) {
  cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDefault);
}
}  // namespace gccl
