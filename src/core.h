#pragma once

#include "comm/comm_info.h"
#include "gccl.h"

namespace gccl {

// Propagate errors up
#define GCCLCHECK(call)             \
  do {                              \
    gcclResult_t res = call;        \
    if (res != gcclSuccess) {       \
      /* Print the back trace*/     \
      LOG(INFO) << "Error " << res; \
      return res; \
    }                               \
  } while (0);

#define GCCLCHECKNORET(call)             \
  do {                              \
    gcclResult_t res = call;        \
    if (res != gcclSuccess) {       \
      /* Print the back trace*/     \
      LOG(FATAL) << "Error " << res; \
    }                               \
  } while (0);

// ENCODE(ENCODE(x)) = x
#define ENCODE(idx) (-(idx)-1)

#define DIVUP(x, y) (((x) + (y)-1) / (y))
#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))

template <typename T>
static void GCCLCalloc(T** ptr, size_t nelem) {
  void* p = malloc(nelem * sizeof(T));
  if (p == NULL) {
    LOG(FATAL) << "Failed to malloc " << nelem * sizeof(T) << " bytes";
    return;
  }
  memset(p, 0, nelem * sizeof(T));
  *ptr = (T*)p;
}

template <typename T>
static void GCCLCallocAndCopy(T** ptr, const std::vector<T>& vec) {
  void *p = malloc(vec.size() * sizeof(T));
  memcpy(p, vec.data(), vec.size() * sizeof(T));
  *ptr = (T*)p;
}
size_t GetDataTypeSize(gcclDataType_t type);

struct CollectiveArgs {
  CommInfo* info;
  void* input;
  int ele_size;
  int feat_size;
  int buffer_size;
  int rank;
  int threads_per_conn;
};

}  // namespace gccl
