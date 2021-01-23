#pragma once

#include <thread>
#include <vector>

#include "gccl.h"

namespace gccl {

#define TRANSPORT_PROXY_FIFO_SIZE 2048

struct gcclProxyArgs {
  int n_stages;
  std::vector<int> max_comm_size; // Per stage max comm size in all stages
  std::vector<int> comm_off;       // Per stage send/recv offset
  int buffer_size;
  int feat_size;
  int n_threads;
  void* resources;
  int active;  // add component before this line -- it is left out during
               // initialization
};

typedef void (*threadFunc_t)(struct gcclProxyArgs*);

struct transportProxyInfo {
  int dev_id;
  pthread_t thread;
  threadFunc_t func;
  volatile int proxyReady;
  struct gcclProxyArgs argsFifo[TRANSPORT_PROXY_FIFO_SIZE];
  volatile uint64_t argsFifoHead;
  volatile uint64_t argsFifoTail;
  pthread_cond_t cond;
  pthread_mutex_t mutex;
};

void CreateProxy(transportProxyInfo** info, threadFunc_t func);

gcclResult_t transportSaveProxy(transportProxyInfo* info, gcclProxyArgs* args);

gcclResult_t transportStartProxy(transportProxyInfo* proxy);

inline void transportProxyIdle(int idle) { sched_yield(); }

}  // namespace gccl
