#pragma once

#include "comm/pattern/comm_pattern.h"
#include "transport.h"

namespace gccl {

template <typename T>
using Vec3D = std::vector<std::vector<std::vector<T>>>;

// (src, dst, stage, node id)
struct GreedyTransferInfo {
  std::vector<Vec3D<int>> tr_ids;
};
struct GreedyCommPatternInfo {
  int n_peers;
  int rank;
  int n_stages;
  int extra_buffer_size;
  int max_feat_size;
  int *max_comm_size,
      *cpu_max_comm_size;  // max communication size for each stage
  int *send_ids;           // flatten 3D array, (stage, peer_id, send_id)
  int *send_off;  // flatten 2D array, (stage_off, peer_id), size is n_stages *
                  // n_peers + 1
  int *recv_ids;
  int *recv_off;
  int *cpu_send_off, *cpu_recv_off;
  int threads_per_conn;

  int n_conn;
  int *conn_peers;
  void *dev_extra_mem;

  int buffer_size;

  SendDevMem **send_dev_mem;  // 1D, peer_id
  RecvDevMem **recv_dev_mem;

  ConnInfo *conn_info;    // 1D
  void **send_resources;  // 1D
  void **recv_resources;  // 1D

  // Proxy
  transportProxyInfo **send_proxy_info;  // Proxy to different host. One proxy
                                         // per remote connection.
  transportProxyInfo **recv_proxy_info;  // Proxy to different host. One proxy
                                         // per remote connection.

  BinStream &serialize(BinStream &stream) const;
  BinStream &deserialize(BinStream &stream);
  void CopyGraphInfoToDev();
  void Print() const;
  int GetMemBytes() const;
};

class GreedyCommPattern : public CommPattern {
 public:
  GreedyCommPattern(const std::vector<int> &dev_topo,
                    const std::vector<ConnType> &conn_type)
      : CommPattern(dev_topo, conn_type) {}
  ~GreedyCommPattern() override {}

  std::vector<CommPatternInfo> BuildCommPatternInfos(
      Config *config, const std::vector<std::map<int, int>> &local_mappings,
      const TransferRequest &req, int nparts) override;

  void SetupConnection(CommPatternInfo *info, Coordinator *coor, int bid,
                       const std::vector<int> &conn_peers) override;
  void StartProxy(Coordinator *coor, CommPatternInfo *info) override;
  void SaveProxy(Coordinator *coor, CommPatternInfo *info, int feat_size,
                 int n_threads, bool forward) override;

 private:
  std::vector<int> conn_peers_;
};
}  // namespace gccl
