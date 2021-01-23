#pragma once

#include "comm/pattern/comm_pattern.h"
#include "transport.h"

namespace gccl {

struct AllToAllCommPatternInfo {
  int n_peers;
  int rank;
  int max_comm_size;  // max communication size for each stage
  int *send_ids;
  int *send_off;
  int *recv_ids;
  int *recv_off;
  int *cpu_send_off, *cpu_recv_off;  // for proxy

  int buffer_size;
  int threads_per_conn, pad;

  SendDevMem **send_dev_mem;
  RecvDevMem **recv_dev_mem;

  ConnInfo *conn_info;
  void **send_resources;
  void **recv_resources;

  // Proxy
  transportProxyInfo **send_proxy_info;  // Proxy to different host. One proxy
                                         // per remote connection.
  transportProxyInfo **recv_proxy_info;  // Proxy to different host. One proxy
                                         // per remote connection.

  BinStream &serialize(BinStream &stream) const;
  BinStream &deserialize(BinStream &stream);
  void CopyGraphInfoToDev();
  void Print() const;
};

class AllToAllCommPattern : public CommPattern {
 public:
  AllToAllCommPattern(const std::vector<int> &dev_topo,
                      const std::vector<ConnType> &conn_type)
      : CommPattern(dev_topo, conn_type) {}
  ~AllToAllCommPattern() override {}

  std::vector<CommPatternInfo> BuildCommPatternInfos(
      Config *config, const std::vector<std::map<int, int>> &local_mappings,
      const TransferRequest &req, int nparts) override;

  void SetupConnection(CommPatternInfo *info, Coordinator *coor, int bid,
                       const std::vector<int> &conn_peers) override;

  void StartProxy(Coordinator *coor, CommPatternInfo *info) override;
  void SaveProxy(Coordinator *coor, CommPatternInfo *info, int feat_size,
                 int n_threads, bool forward) override;
};
}  // namespace gccl
