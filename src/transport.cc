#include "transport.h"

#include <thread>

#include "comm/comm_info.h"
#include "core.h"

namespace gccl {

static void FifoPullArgs(struct transportProxyInfo* info,
                         struct gcclProxyArgs* args) {
  struct gcclProxyArgs* fifoArgs =
      info->argsFifo + (info->argsFifoHead % TRANSPORT_PROXY_FIFO_SIZE);
  pthread_mutex_lock(&info->mutex);
  while (fifoArgs->active == 0) pthread_cond_wait(&info->cond, &info->mutex);
  __sync_synchronize();
  memcpy(args, fifoArgs, sizeof(struct gcclProxyArgs));
  __sync_synchronize();
  fifoArgs->active = 0;
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
  info->argsFifoHead++;
}

static void SetProxyReady(struct transportProxyInfo* info) {
  pthread_mutex_lock(&info->mutex);
  info->proxyReady = 1;
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
}

static void WaitProxyReady(struct transportProxyInfo* info) {
  pthread_mutex_lock(&info->mutex);
  while (info->proxyReady == 0) pthread_cond_wait(&info->cond, &info->mutex);
  pthread_mutex_unlock(&info->mutex);
}

void* persistentThread(void* opaqueInfo) {
  struct transportProxyInfo* info = (struct transportProxyInfo*)opaqueInfo;
  // We need to initialize the context before launching any NCCL cuda kernel,
  // otherwise we would create it during the first cudaMemcpyAsync inside the
  // proxy function and that would cause a deadlock
  cudaSetDevice(info->dev_id);
  // Signal the main thread the context is created and it can proceed.
  SetProxyReady(info);
  while (1) {
    struct gcclProxyArgs args;
    FifoPullArgs(info, &args);
    if (args.active == -1) {
      // Main thread asked to stop
      return NULL;
    }
    info->func(&args);
  }
}

void CreateProxy(transportProxyInfo** proxy_info, threadFunc_t func) {
  GCCLCalloc(proxy_info, 1);
  transportProxyInfo* info = *proxy_info;
  // info->comm = comm;
  info->cond = PTHREAD_COND_INITIALIZER;
  info->mutex = PTHREAD_MUTEX_INITIALIZER;
  info->func = func;
  info->argsFifoHead = info->argsFifoTail = 0;
  info->proxyReady = 0;
  pthread_create(&info->thread, NULL, persistentThread, info);
  // Wait for thread to initialize its CUDA context.
  WaitProxyReady(info);
}
static void FifoPushArgs(struct transportProxyInfo* info) {
  if (info == NULL) return;

  struct gcclProxyArgs* fifoArgs =
      info->argsFifo + ((info->argsFifoTail - 1) % TRANSPORT_PROXY_FIFO_SIZE);
  if (fifoArgs->active == 0) return;

  pthread_mutex_lock(&info->mutex);
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
}

static struct gcclProxyArgs* FifoGetNextArgs(struct transportProxyInfo* info) {
  if (info == NULL) return NULL;
  struct gcclProxyArgs* fifoArgs =
      info->argsFifo + (info->argsFifoTail % TRANSPORT_PROXY_FIFO_SIZE);
  pthread_mutex_lock(&info->mutex);
  while (fifoArgs->active == 1) pthread_cond_wait(&info->cond, &info->mutex);
  pthread_mutex_unlock(&info->mutex);
  info->argsFifoTail++;
  return fifoArgs;
}

gcclResult_t transportSaveProxy(transportProxyInfo* info, gcclProxyArgs* args) {
  if(info == nullptr) return gcclSuccess;
  struct gcclProxyArgs* fifoArgs = FifoGetNextArgs(info);
  __sync_synchronize();
  *fifoArgs = *args;
  // memcpy(fifoArgs, args, sizeof(struct gcclProxyArgs));
  __sync_synchronize();
  fifoArgs->active = 1;
  return gcclSuccess;
}
gcclResult_t transportStartProxy(transportProxyInfo* proxy) {
  FifoPushArgs(proxy);
  return gcclSuccess;
}

}  // namespace gccl
