#pragma once

#include "conn/connection.h"
#include "conn/net.h"
#include "transport.h"

namespace gccl {

struct NetExchangeConnInfo {
  gcclNetHandle_t netHandle;
};

struct netSendResources {
  void* netSendComm;
  struct SendDevMem* hostSendMem;
  struct RecvDevMem* hostRecvMem;
  struct SendDevMem* devHostSendMem;
  struct RecvDevMem* devHostRecvMem;
  struct SendDevMem* hostDevMem;
  int netDev;
  bool cudaSupport;
  struct RecvDevMem* devNetMem;
};

struct netRecvResources {
  void* netListenComm;
  void* netRecvComm;
  struct SendDevMem* hostSendMem;
  struct RecvDevMem* hostRecvMem;
  struct SendDevMem* devHostSendMem;
  struct RecvDevMem* devHostRecvMem;
  struct RecvDevMem* hostDevMem;
  struct RecvDevMem* devNetMem;
  int netDev;
  bool cudaSupport;
};
class NetConnection : public Connection {
 public:
  NetConnection(const ProcInfo& my_info, const ProcInfo& peer_info, int bid)
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

 private:
  int n_net_devs_;
};

void NetSendProxy(gcclProxyArgs* args);
void NetRecvProxy(gcclProxyArgs* args);

}  // namespace gccl