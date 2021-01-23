#include "conn/net_connection.h"

#include "conn/net.h"
#include "core.h"
#include "gpu/common.h"
#include "param.h"

namespace gccl {

int getDev(int ringId, int nDev, int* scores) {
  int maxScore = 0;
  for (int d = 0; d < nDev; d++)
    if (scores[d] > maxScore) maxScore = scores[d];
  int skip = ringId + 1;
  while (skip) {
    for (int d = 0; d < nDev; d++) {
      if (scores[d] == maxScore) {
        skip--;
        if (skip == 0) return d;
      }
    }
  }
  return 0;
}

void NetConnection::SendSetup(SendDevMem** send_dev_mem, void** send_resources,
                              int buffer_size, ConnInfo* conn_info,
                              ExchangeConnInfo* ex_info) {
  struct netSendResources* resources;
  GCCLCalloc(&resources, 1);
  *send_resources = resources;
  int n_net_devs, *score;
  gcclNetDevices(&n_net_devs, &score);
  resources->netDev = getDev(bid_, n_net_devs, score);
  resources->cudaSupport = false;
  int gdrReadParam = GetEnvParam("NET_GDR_READ", 1);

  bool enableGdrRead = gdrReadParam > 0;

  if (enableGdrRead) {
    int flags;
    GCCLCHECKNORET(gcclNetPtrSupport(resources->netDev, &flags));
    if (flags & GCCL_PTR_CUDA) resources->cudaSupport = true;
  }
  if (resources->cudaSupport)
    DLOG(INFO) << "Net: enabling net device " << resources->netDev
              << " to read from rank " << my_info_.rank;
  int size = offsetof(struct RecvDevMem, buff) + buffer_size;
  if (resources->cudaSupport) {
    GCCLCudaMalloc((char**)(&resources->devNetMem), size);
  }
  GCCLCudaHostAlloc((void**)&resources->hostRecvMem,
                    (void**)&resources->devHostRecvMem, size);

  GCCLCudaHostAlloc((void**)&resources->hostSendMem,
                    (void**)&resources->devHostSendMem, size);
}

void NetConnection::RecvSetup(RecvDevMem** recv_dev_mem, void** recv_resources,
                              int buffer_size, ConnInfo* conn_info,
                              ExchangeConnInfo* ex_info) {
  struct netRecvResources* resources;
  GCCLCalloc(&resources, 1);
  *recv_resources = resources;

  int n_net_devs, *score;
  gcclNetDevices(&n_net_devs, &score);
  resources->netDev = getDev(bid_, n_net_devs, score);

  int flags;
  GCCLCHECKNORET(gcclNetPtrSupport(resources->netDev, &flags));
  resources->cudaSupport = (flags & GCCL_PTR_CUDA) ? true : false;

  int sendSize = sizeof(struct SendDevMem);

  GCCLCudaHostAlloc((void**)&resources->hostSendMem,
                    (void**)&resources->devHostSendMem, sendSize);

  int recvSize = offsetof(struct RecvDevMem, buff) + buffer_size;
  if (resources->cudaSupport) {
    GCCLCudaMalloc((void**)&resources->devNetMem, recvSize);
  }
  GCCLCudaHostAlloc((void**)&resources->hostRecvMem,
                    (void**)&resources->devHostRecvMem, recvSize);

  struct NetExchangeConnInfo* info = (struct NetExchangeConnInfo*)ex_info;
  GCCLCHECKNORET(gcclNetListen(resources->netDev, &info->netHandle,
                          &resources->netListenComm));
}

void NetConnection::SendConn(ConnInfo* conn_info, void* send_resources,
                             int buffer_size, ExchangeConnInfo* peer_ex_info) {
  struct netSendResources* resources = (netSendResources*)send_resources;
  conn_info->my_substage_done = &resources->devHostSendMem->substage_done;

  if (resources->cudaSupport) {
    conn_info->next_recv_buff = resources->devNetMem->buff;
  } else {
    conn_info->next_recv_buff = &resources->devHostRecvMem->buff;
  }
  conn_info->next_stage_ready = &resources->devHostRecvMem->stage_ready;
  conn_info->next_substage_ready = &resources->devHostRecvMem->substage_ready;

  // Connect to remote peer
  struct NetExchangeConnInfo* info = (struct NetExchangeConnInfo*)peer_ex_info;
  GCCLCHECKNORET(gcclNetConnect(resources->netDev, info->netHandle,
                           &resources->netSendComm));
}

void NetConnection::RecvConn(ConnInfo* conn_info, void* recv_resources,
                             ExchangeConnInfo* peer_ex_info) {
  struct netRecvResources* resources = (netRecvResources*)recv_resources;
  conn_info->my_stage_ready = &resources->devHostRecvMem->stage_ready;
  conn_info->my_substage_ready = &resources->devHostRecvMem->substage_ready;

  conn_info->prev_substage_done = &resources->devHostSendMem->substage_done;
  if (resources->cudaSupport) {
    conn_info->my_recv_buff = resources->devNetMem->buff;
  } else {
    conn_info->my_recv_buff = resources->devHostRecvMem->buff;
  }

  GCCLCHECKNORET(gcclNetAccept(resources->netListenComm, &resources->netRecvComm));
  GCCLCHECKNORET(gcclNetCloseListen(resources->netListenComm));
}

void NetSendProxy(gcclProxyArgs* args) {
  int n_stages = args->n_stages;
  int buffer_size = args->buffer_size;
  int feat_size = args->feat_size;
  netSendResources* resources = (netSendResources*)args->resources;
  void* local_buff = resources->cudaSupport ? resources->devNetMem->buff
                                            : resources->hostRecvMem->buff;
  volatile uint64_t* next_ready = &resources->hostRecvMem->stage_ready;
  volatile uint64_t* next_substage_ready =
      &resources->hostRecvMem->substage_ready;
  volatile uint64_t* my_substage_done = &resources->hostSendMem->substage_done;
  int ptr_type = resources->cudaSupport ? GCCL_PTR_CUDA : GCCL_PTR_HOST;

  uint64_t flag_end = (~0ull >> 1) + 1;  // FIXME
  DLOG(INFO) << "Net send proxy:";
  DLOG(INFO) << " # n_stages: " << args->n_stages;
  DLOG(INFO) << " # feat_size: " << feat_size;
  DLOG(INFO) << " # n_threads: " << args->n_threads;
  DLOG(INFO) << " # max_comm_size: " << VecToString(args->max_comm_size);
  DLOG(INFO) << " # comm_off: " << VecToString(args->comm_off);

  for (int stage = 0; stage < n_stages; ++stage) {
    void* request = nullptr;
    uint64_t head = 0, tail = 0;

    int type_size = 4;  // FIXME (FIX_TYPE_SIZE)
    int comm_size = args->comm_off[stage];
    int max_comm_size = args->max_comm_size[stage];
    int per_substage_n_ele = buffer_size /
                             (feat_size * type_size * args->n_threads) *
                             args->n_threads;
    int n_substages = DIVUP(max_comm_size, per_substage_n_ele);
    int per_substage_size = feat_size * type_size * per_substage_n_ele;
    int remaining_comm_size = comm_size * feat_size * type_size;

    *next_ready = stage + 1;
    tail++;
    *next_substage_ready = tail;
    DLOG(INFO) << "Send proxy start for stage " << stage << " with substages: " << n_substages << " " << 
      "Comm size " << comm_size;
    while (tail <= n_substages) {
      int idle = 1;
      if (head >= tail) {  // expect head == tail
        int done;
        GCCLCHECKNORET(gcclNetTest(request, &done, NULL));
        DLOG(INFO) << "Checking whether send done";
        if (done) {
          DLOG(INFO) << "Send proxy done, set next_substage_ready to " << tail + 1;
          tail++;
          *next_substage_ready = tail;
          idle = 0;
        }
      }
      if (head + 1 == *my_substage_done) {
        int size = std::min(per_substage_size, remaining_comm_size);
        remaining_comm_size -= per_substage_size;
        if (remaining_comm_size < 0) {
          remaining_comm_size = 0;
        }
        if(size == 0) size = 1;
        DLOG(INFO) << "Send proxy send with head " << head << " with size " << size;
        GCCLCHECKNORET(gcclNetIsend(resources->netSendComm, local_buff, size,
                               ptr_type, &request));
        DLOG(INFO) << "Async send done";
        head++;
        idle = 0;
      }
      if (idle) transportProxyIdle(idle);
    }
    *next_substage_ready = flag_end;
    while(*my_substage_done != flag_end) {
      sched_yield();
    }
  }
}

void NetRecvProxy(gcclProxyArgs* args) {
  int n_stages = args->n_stages;
  int buffer_size = args->buffer_size;
  int feat_size = args->feat_size;
  netRecvResources* resources = (netRecvResources*)args->resources;

  DLOG(INFO) << "resources " << resources;
  DLOG(INFO) << "resources->cudaSupport? " << resources->cudaSupport;
  void* local_buff = resources->cudaSupport ? resources->devNetMem->buff
                                            : resources->devHostRecvMem->buff;
  volatile uint64_t* prev_done = &resources->hostSendMem->substage_done;
  volatile uint64_t* substage_ready = &resources->hostRecvMem->substage_ready;
  int ptr_type = resources->cudaSupport ? GCCL_PTR_CUDA : GCCL_PTR_HOST;

  uint64_t flag_end = (~0ull >> 1) + 1;  // FIXME
  DLOG(INFO) << "Net Recv proxy:";
  DLOG(INFO) << " # n_stages: " << args->n_stages;
  DLOG(INFO) << " # feat_size: " << feat_size;
  DLOG(INFO) << " # n_threads: " << args->n_threads;
  DLOG(INFO) << " # max_comm_size: " << VecToString(args->max_comm_size);
  DLOG(INFO) << " # comm_off: " << VecToString(args->comm_off);

  for (int stage = 0; stage < n_stages; ++stage) {
    void* request;

    int type_size = 4;  // FIXME (FIX_TYPE_SIZE)
    int comm_size = args->comm_off[stage];
    int max_comm_size = args->max_comm_size[stage];
    int per_substage_n_ele = buffer_size /
                             (feat_size * type_size * args->n_threads) *
                             args->n_threads;
    int n_substages = DIVUP(max_comm_size, per_substage_n_ele);
    int per_substage_size = feat_size * type_size * per_substage_n_ele;
    int remaining_comm_size = comm_size * feat_size * type_size;

    DLOG(INFO) << "Recv proxy start for stage " << stage << " with substages: " << n_substages;
    int head = 0, tail = 0;
    while (head < n_substages) {
      int idle = 1;
      if (tail + 1 == *substage_ready) {
        int size = std::min(per_substage_size, remaining_comm_size);
        remaining_comm_size -= per_substage_size;
        if (remaining_comm_size < 0) {
          remaining_comm_size = 0;
        }
        if(size == 0) size = 1;
        DLOG(INFO) << "Recv proxy try to Irecv with tail " << tail + 1;
        GCCLCHECKNORET(gcclNetIrecv(resources->netRecvComm, local_buff, size,
                               ptr_type, &request));
        tail++;
        idle = 0;
      }
      if (head < tail) {
        int done;
        GCCLCHECKNORET(gcclNetTest(request, &done, NULL));
        if (done) {
          head++;
          DLOG(INFO) << "Recv proxy done with head " << head;
          *prev_done = head;
          idle = 0;
        }
      }
      if (idle) transportProxyIdle(idle);
    }
    DLOG(INFO) << "Recv proxy finished stage " << stage;
    while(*substage_ready != flag_end) {
      sched_yield();
    }
    *prev_done = flag_end;
  }
}

}  // namespace gccl
