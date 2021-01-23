#include "shm_connection.h"

#include "comm/pattern/comm_pattern.h"
#include "glog/logging.h"
#include "gpu/common.h"
#include "shm.h"

namespace gccl {

#define MAX_SHM_NAME_LEN 1024

void ShmConnection::SendSetup(SendDevMem** send_dev_mem, void** send_resources,
                              int buffer_size, ConnInfo* conn_info,
                              ExchangeConnInfo* ex_info) {
  char shm_name[MAX_SHM_NAME_LEN];
  ShmExchangeConnInfo shm_conn_info;
  ShmSendResources* res = new ShmSendResources;
  *send_resources = res;
  sprintf(shm_name, "gccl-shm-send-%d-%d-%d", my_info_.rank, peer_info_.rank,
          bid_);
  int shm_size = sizeof(SendDevMem);
  gccl_shm_open(shm_name, shm_size, (void**)&res->host_mem,
                (void**)&res->send_dev_mem, 1);

  shm_conn_info.rank = my_info_.rank;
  shm_conn_info.peer_rank = peer_info_.rank;
  shm_conn_info.bid = bid_;

  *send_dev_mem = res->send_dev_mem;
  conn_info->my_substage_done = &res->send_dev_mem->substage_done;

  static_assert(sizeof(ShmExchangeConnInfo) <= sizeof(ExchangeConnInfo),
                "ShmExchangeConnInfo too large");
  memcpy(ex_info, &shm_conn_info, sizeof(shm_conn_info));
  DLOG(INFO) << "Block " << bid_ << " send setup with shared memory";
}

void ShmConnection::RecvSetup(RecvDevMem** recv_dev_mem, void** recv_resources,
                              int buffer_size, ConnInfo* conn_info,
                              ExchangeConnInfo* ex_info) {
  ShmExchangeConnInfo shm_conn_info;
  ShmRecvResources* res = new ShmRecvResources;
  *recv_resources = res;

  char shm_name[MAX_SHM_NAME_LEN];
  sprintf(shm_name, "gccl-shm-recv-%d-%d-%d", my_info_.rank, peer_info_.rank,
          bid_);
  int shm_size = sizeof(RecvDevMem) + buffer_size;

  gccl_shm_open(shm_name, shm_size, (void**)&res->host_mem,
                (void**)&res->recv_dev_mem, 1);

  shm_conn_info.rank = my_info_.rank;
  shm_conn_info.peer_rank = peer_info_.rank;
  shm_conn_info.bid = bid_;

  *recv_dev_mem = res->recv_dev_mem;
  conn_info->my_stage_ready = &res->recv_dev_mem->stage_ready;
  conn_info->my_substage_ready = &res->recv_dev_mem->substage_ready;
  conn_info->my_recv_buff = res->recv_dev_mem->buff;

  static_assert(sizeof(ShmExchangeConnInfo) <= sizeof(ExchangeConnInfo),
                "ShmExchangeConnInfo too large");
  memcpy(ex_info, &shm_conn_info, sizeof(shm_conn_info));
  DLOG(INFO) << "Block " << bid_ << " recv setup with shared memory";
}

/*
void ShmConnection::RecvSetup(CommPatternInfo* info,
                              ExchangeConnInfo* conn_info) {
  int buffer_size = info->buffer_size;
  int max_feat_size = info->max_feat_size;
  int extra_mem_size = info->extra_buff_size * max_feat_size * 4;
  extra_mem_size = std::max(extra_mem_size, 1);
  GCCLCudaMalloc((char**)&info->dev_extra_mem, extra_mem_size);

  ShmExchangeConnInfo shm_conn_info;
  ShmRecvResources* res = new ShmRecvResources;
  info->recv_resources = res;

  char shm_name[MAX_SHM_NAME_LEN];
  sprintf(shm_name, "gccl-shm-recv-%d-%d-%d", my_info_.rank, my_info_.dev_id,
          bid_);
  int shm_size = sizeof(RecvDevMem) + buffer_size;

  gccl_shm_open(shm_name, shm_size, (void**)&res->host_mem,
                (void**)&res->recv_dev_mem, 1);

  shm_conn_info.rank = my_info_.rank;
  shm_conn_info.dev_id = my_info_.dev_id;
  shm_conn_info.bid = bid_;

  info->recv_dev_mem = res->recv_dev_mem;
  info->conn_info.my_stage_ready = &res->recv_dev_mem->stage_ready;
  info->conn_info.my_substage_ready = &res->recv_dev_mem->substage_ready;

  static_assert(sizeof(ShmExchangeConnInfo) <= sizeof(ExchangeConnInfo),
                "ShmExchangeConnInfo too large");
  memcpy(conn_info, &shm_conn_info, sizeof(shm_conn_info));
  DLOG(INFO) << "Block " << bid_ << " recv setup with shared memory";
}
*/

void ShmConnection::SendConn(ConnInfo* conn_info, void* send_resources,
                             int buffer_size, ExchangeConnInfo* peer_ex_info) {
  ShmExchangeConnInfo* peer_info = (ShmExchangeConnInfo*)peer_ex_info;
  char shm_name[MAX_SHM_NAME_LEN];
  sprintf(shm_name, "gccl-shm-recv-%d-%d-%d", peer_info->rank,
          peer_info->peer_rank, peer_info->bid);
  auto* res = (ShmSendResources*)send_resources;
  int shm_size = sizeof(RecvDevMem) + buffer_size;

  gccl_shm_open(shm_name, shm_size, (void**)&res->rem_host_mem,
                (void**)&res->rem_recv_dev_mem, 0);
  gccl_shm_unlink(shm_name);

  conn_info->next_stage_ready = &res->rem_recv_dev_mem->stage_ready;
  conn_info->next_substage_ready = &res->rem_recv_dev_mem->substage_ready;
  conn_info->next_recv_buff = &res->rem_recv_dev_mem->buff;

  DLOG(INFO) << "Block " << bid_ << " send connect shared memory";
}

void ShmConnection::RecvConn(ConnInfo* conn_info, void* recv_resources,
                             ExchangeConnInfo* peer_ex_info) {
  ShmExchangeConnInfo* peer_info = (ShmExchangeConnInfo*)peer_ex_info;
  char shm_name[MAX_SHM_NAME_LEN];
  sprintf(shm_name, "gccl-shm-send-%d-%d-%d", peer_info->rank,
          peer_info->peer_rank, peer_info->bid);
  auto* res = (ShmRecvResources*)recv_resources;
  int shm_size = sizeof(SendDevMem);
  gccl_shm_open(shm_name, shm_size, (void**)&res->rem_host_mem,
                (void**)&res->rem_send_dev_mem, 0);
  gccl_shm_unlink(shm_name);

  conn_info->prev_substage_done = &res->rem_send_dev_mem->substage_done;

  DLOG(INFO) << "Block " << bid_ << " recv connect shared memory";
}

}  // namespace gccl
