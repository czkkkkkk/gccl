#pragma once

#include "coordinator.h"

namespace gccl {

#define MAX_CONN_INFO_SIZE 256
#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096

struct SendDevMem {
  union {
    struct {
      uint64_t substage_done;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
    };
    char pad2[MEM_ALIGN];
  };
};

struct __align__(16) RecvDevMem {
  union {
    struct {
      uint64_t stage_ready;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
      uint64_t substage_ready;
      char pad2[CACHE_LINE_SIZE - sizeof(uint64_t)];
    };
    char pad3[MEM_ALIGN];
  };
  char buff[1];  // Actually larger than that
};

struct ConnInfo {
  uint64_t* my_stage_ready;
  uint64_t* next_stage_ready;

  uint64_t* next_substage_ready;
  uint64_t* my_substage_ready;
  uint64_t* prev_substage_done;
  uint64_t* my_substage_done;

  void* next_recv_buff;
  void* my_recv_buff;
};

enum ConnType { P2P, SHM, IB };

struct ExchangeConnInfo {
  char info[MAX_CONN_INFO_SIZE];
};

ConnType GetConnType(const ProcInfo& my_info, const ProcInfo& peer_info);

class Connection {
 public:
  Connection(const ProcInfo& my_info, const ProcInfo& peer_info, int bid)
      : my_info_(my_info), peer_info_(peer_info), bid_(bid) {}

  // FIXME: We may not initialize send_dev_mem and recv_dev_mem here
  virtual void SendSetup(SendDevMem** send_dev_mem, void** send_resources,
                         int buffer_size, ConnInfo* conn_info,
                         ExchangeConnInfo* ex_info) = 0;
  virtual void RecvSetup(RecvDevMem** recv_dev_mem, void** recv_resources,
                         int buffer_size, ConnInfo* conn_info,
                         ExchangeConnInfo* ex_info) = 0;
  virtual void SendConn(ConnInfo* conn_info, void* send_resources,
                        int buffer_size, ExchangeConnInfo* peer_ex_info) = 0;

  virtual void RecvConn(ConnInfo* conn_info, void* recv_resources,
                        ExchangeConnInfo* peer_ex_info) = 0;

 protected:
  ProcInfo my_info_, peer_info_;
  int bid_;
};

class ConnectionFactory {
 public:
  static std::shared_ptr<Connection> GetConnection(ConnType type,
                                                   const ProcInfo& my_info,
                                                   const ProcInfo& peer_info,
                                                   int bid);
};

}  // namespace gccl
