#pragma once

#include <cuda_runtime.h>

#include "connection.h"

namespace gccl {

struct P2pExchangeConnInfo {
  cudaIpcMemHandle_t dev_ipc;
};

class P2pConnection : public Connection {
 public:
  P2pConnection(const ProcInfo& my_info, const ProcInfo& peer_info, int bid)
      : Connection(my_info, peer_info, bid) {}

  void SendSetup(SendDevMem** send_dev_mem, void** send_resources,
                 int buffer_size, ConnInfo* conn_info,
                 ExchangeConnInfo* ex_info) override;

  void RecvSetup(RecvDevMem** recv_dev_mem, void** recv_resources,
                 int buffer_size, ConnInfo* conn_info,
                 ExchangeConnInfo* ex_info) override;

  void SendConn(ConnInfo* conn_info, void* send_resources, int buffer_size,
                ExchangeConnInfo* peer_ex_info) override;

  void RecvConn(ConnInfo* conn_info, void* recv_resources,
                ExchangeConnInfo* peer_ex_info) override;
};

}  // namespace gccl
