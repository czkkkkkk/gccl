/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "glog/logging.h"

#include "utils.h"

namespace gccl {

static void gccl_shm_open(const char* shmname, int shmsize, void** shmPtr,
                          void** devShmPtr, int create) {
  *shmPtr = nullptr;
  int fd = shm_open(shmname, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (fd == -1) {
    LOG(FATAL) << "Cannot open shared memory " << shmname << " : "
               << strerror(errno);
    return;
  }

  if (create) {
    int res = posix_fallocate(fd, 0, shmsize);
    if (res != 0) {
      LOG(FATAL) << "Unable to allocate shared memory (" << shmsize
                 << " bytes) : " << strerror(res);
      shm_unlink(shmname);
      close(fd);
      return;
    }
  }

  void* ptr = mmap(NULL, shmsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (ptr == MAP_FAILED) {
    LOG(FATAL) << "failure in mmap of " << shmname << " (size " << shmsize
               << ") : " << strerror(errno);
    shm_unlink(shmname);
    return;
  }
  if (create) {
    memset(ptr, 0, shmsize);
  }

  cudaError_t e;
  if ((e = cudaHostRegister(ptr, shmsize, cudaHostRegisterMapped)) !=
      cudaSuccess) {
    LOG(FATAL) << "failed to register host buffer " << ptr << " : "
               << cudaGetErrorString(e);
    if (create) shm_unlink(shmname);
    munmap(ptr, shmsize);
    return;
  }

  if ((e = cudaHostGetDevicePointer(devShmPtr, ptr, 0)) != cudaSuccess) {
    LOG(FATAL) << "failed to get device pointer for local shmem " << ptr
               << " : " << cudaGetErrorString(e);
    if (create) shm_unlink(shmname);
    munmap(ptr, shmsize);
    return;
  }
  *shmPtr = ptr;
}
static bool gccl_shm_unlink(const char* shmname) {
  if (shmname != NULL) SYSCHECK(shm_unlink(shmname), "shm_unlink");
  return true;
}

}  // namespace gccl
