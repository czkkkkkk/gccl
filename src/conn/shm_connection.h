#pragma once

#include "connection.h"

namespace gccl {

struct ShmExchangeConnInfo {
  int rank, peer_rank, bid;
};

struct ShmSendResources {
  RecvDevMem* rem_host_mem;
  RecvDevMem* rem_recv_dev_mem;
  SendDevMem* host_mem;
  SendDevMem* send_dev_mem;
};
struct ShmRecvResources {
  SendDevMem* rem_host_mem;
  SendDevMem* rem_send_dev_mem;
  RecvDevMem* host_mem;
  RecvDevMem* recv_dev_mem;
};

class ShmConnection : public Connection {
 public:
  ShmConnection(const ProcInfo& my_info, const ProcInfo& peer_info, int bid)
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
