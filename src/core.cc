#include "core.h"

#include "glog/logging.h"

namespace gccl {

size_t GetDataTypeSize(gcclDataType_t type) {
  switch (type) {
    case gcclFloat:
      return sizeof(float);
    case gcclInt:
      return sizeof(int);
    default:
      CHECK(false) << "Unrecoginized type";
  }
}

}  // namespace gccl