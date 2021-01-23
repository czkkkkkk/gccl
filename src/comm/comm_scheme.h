#pragma once

#include <memory>
#include <vector>

#include "comm/pattern/comm_pattern.h"

#define MAX_BLOCKS 64

namespace gccl {

struct CommScheme {
  CommPatternInfo comm_pattern_infos[MAX_BLOCKS];
  int n_blocks;

  // Only serialize graph info
  BinStream &serialize(BinStream &stream) const;
  BinStream &deserialize(BinStream &stream);
  void CopyGraphInfoToDev();
  void Print() const;
};

void CommSchemeSetupConnection(
    CommScheme *scheme, Coordinator *coor,
    const std::vector<std::shared_ptr<CommPattern>> &patterns, const std::vector<int>& conn_peers);

}  // namespace gccl
