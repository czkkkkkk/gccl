#pragma once

#include "base/bin_stream.h"
#include "comm_scheme.h"

namespace gccl {

struct CommInfo {
  // When move the comm info to GPU, we want to allocate contiguous memory to
  // them.
  CommScheme allgather_scheme;
  CommInfo *dev_info;
  int n_conn_peers;
  int *conn_peers; // On CPU

  // Only serialize graph info
  BinStream &serialize(BinStream &stream) const;
  BinStream &deserialize(BinStream &stream);
  void CopyGraphInfoToDev();
  void Print() const;
};

void CommInfoSetupConnection(
    CommInfo *info, Coordinator *coor,
    const std::vector<std::shared_ptr<CommPattern>> &patterns);

void CopyCommInfoToDev(CommInfo *info);

}  // namespace gccl
