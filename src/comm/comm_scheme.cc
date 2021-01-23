#include "comm/comm_scheme.h"

#include "glog/logging.h"

namespace gccl {

BinStream &CommScheme::serialize(BinStream &stream) const {
  stream << n_blocks;
  for (int i = 0; i < n_blocks; ++i) {
    comm_pattern_infos[i].serialize(stream);
  }
  return stream;
}
BinStream &CommScheme::deserialize(BinStream &stream) {
  stream >> n_blocks;
  for (int i = 0; i < n_blocks; ++i) {
    comm_pattern_infos[i].deserialize(stream);
  }
  return stream;
}

void CommSchemeSetupConnection(
    CommScheme *scheme, Coordinator *coor,
    const std::vector<std::shared_ptr<CommPattern>> &patterns, const std::vector<int>& conn_peers) {
  CHECK_LE(scheme->n_blocks, MAX_BLOCKS);
  for (int i = 0; i < scheme->n_blocks; ++i) {
    patterns[i]->SetupConnection(&scheme->comm_pattern_infos[i], coor, i, conn_peers);
  }
}

void CommScheme::CopyGraphInfoToDev() {
  for (int i = 0; i < n_blocks; ++i) {
    comm_pattern_infos[i].CopyGraphInfoToDev();
  }
}

void CommScheme::Print() const {
  int mem_bytes = 0;
  for (int i = 0; i < n_blocks; ++i) {
    DLOG(INFO) << "Block " << i << " :";
    comm_pattern_infos[i].Print();
    mem_bytes += comm_pattern_infos[i].GetMemBytes();
  }
  LOG(INFO) << "Memory overhead for communication information is " << mem_bytes / 1e9 << " GB";
}

}  // namespace gccl
