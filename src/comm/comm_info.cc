#include "comm_info.h"

#include "glog/logging.h"

#include "gpu/common.h"

namespace gccl {

BinStream &CommInfo::serialize(BinStream &stream) const {
  stream << n_conn_peers << std::vector<int>(conn_peers, conn_peers + n_conn_peers);
  return allgather_scheme.serialize(stream);
}

BinStream &CommInfo::deserialize(BinStream &stream) {
  std::vector<int> vconn_peers;
  stream >> n_conn_peers >> vconn_peers;
  CopyVectorToRawPtr(&conn_peers, vconn_peers);
  return allgather_scheme.deserialize(stream);
}

void CommInfo::CopyGraphInfoToDev() { allgather_scheme.CopyGraphInfoToDev(); }

void CommInfo::Print() const { allgather_scheme.Print(); }

void CommInfoSetupConnection(
    CommInfo *info, Coordinator *coor,
    const std::vector<std::shared_ptr<CommPattern>> &patterns) {
  auto conn_peers = std::vector<int>(info->conn_peers, info->conn_peers + info->n_conn_peers);
  CommSchemeSetupConnection(&info->allgather_scheme, coor, patterns, conn_peers);
}

void CopyCommInfoToDev(CommInfo *info) {
  GCCLMallocAndCopy(&info->dev_info, info, 1);
}

}  // namespace gccl
