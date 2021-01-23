#include "p2p_connection.h"

#include <cstring>

#include "glog/logging.h"

#include "comm/pattern/comm_pattern.h"
#include "gpu/common.h"

namespace gccl {

// For debug
uint64_t HashIpcHandle(cudaIpcMemHandle_t handle) {
  uint64_t ret = 0;
  for (int i = 0; i < CUDA_IPC_HANDLE_SIZE; ++i) {
    ret = ret * 233 + handle.reserved[i];
  }
  return ret;
}

void P2pConnection::SendSetup(SendDevMem** send_dev_mem, void** send_resources,
                              int buffer_size, ConnInfo* conn_info,
                              ExchangeConnInfo* ex_info) {
  GCCLCudaMalloc(send_dev_mem, 1);
  conn_info->my_substage_done = &(*send_dev_mem)->substage_done;

  P2pExchangeConnInfo p2p_info;
  cudaIpcGetMemHandle(&p2p_info.dev_ipc, (void*)(*send_dev_mem));
  static_assert(sizeof(P2pExchangeConnInfo) <= sizeof(ExchangeConnInfo),
                "P2P exchange connection info too large");
  memcpy(ex_info, &p2p_info, sizeof(P2pExchangeConnInfo));

  DLOG(INFO) << "Block " << bid_ << " send setup p2p";
}

void P2pConnection::RecvSetup(RecvDevMem** recv_dev_mem, void** recv_resources,
                              int buffer_size, ConnInfo* conn_info,
                              ExchangeConnInfo* ex_info) {
  int dev_mem_size = offsetof(RecvDevMem, buff) + buffer_size;
  DLOG(INFO) << "buffer size is " << buffer_size << " dev mem size "
            << dev_mem_size;
  GCCLCudaMalloc((char**)recv_dev_mem, dev_mem_size);

  conn_info->my_stage_ready = &(*recv_dev_mem)->stage_ready;
  conn_info->my_substage_ready = &(*recv_dev_mem)->substage_ready;
  conn_info->my_recv_buff = (*recv_dev_mem)->buff;

  P2pExchangeConnInfo p2p_info;
  cudaIpcGetMemHandle(&p2p_info.dev_ipc, (void*)(*recv_dev_mem));
  static_assert(sizeof(P2pExchangeConnInfo) <= sizeof(ExchangeConnInfo),
                "P2P exchange connection info too large");
  memcpy(ex_info, &p2p_info, sizeof(P2pExchangeConnInfo));

  DLOG(INFO) << "Block " << bid_ << " recv setup p2p";
}

void P2pConnection::SendConn(ConnInfo* conn_info, void* send_resources,
                             int buffer_size, ExchangeConnInfo* peer_ex_info) {
  P2pExchangeConnInfo* peer_info = (P2pExchangeConnInfo*)peer_ex_info;
  RecvDevMem* ptr;
  CUDACHECK(cudaIpcOpenMemHandle((void**)&ptr, peer_info->dev_ipc,
                                 cudaIpcMemLazyEnablePeerAccess));
  conn_info->next_stage_ready = &ptr->stage_ready;
  conn_info->next_substage_ready = &ptr->substage_ready;
  conn_info->next_recv_buff = &ptr->buff;

  DLOG(INFO) << "Block " << bid_ << " send connect p2p";
}

void P2pConnection::RecvConn(ConnInfo* conn_info, void* recv_resources,
                             ExchangeConnInfo* peer_ex_info) {
  P2pExchangeConnInfo* peer_info = (P2pExchangeConnInfo*)peer_ex_info;
  SendDevMem* ptr;
  CUDACHECK(cudaIpcOpenMemHandle((void**)&ptr, peer_info->dev_ipc,
                                 cudaIpcMemLazyEnablePeerAccess));
  conn_info->prev_substage_done = &ptr->substage_done;

  DLOG(INFO) << "Block " << bid_ << " recv connect p2p";
}

}  // namespace gccl
