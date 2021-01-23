#include "cuda_runtime.h"

namespace gccl {

void GCCLSetCudaDevice(int dev_id) { cudaSetDevice(dev_id); }

}  // namespace gccl