/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include "conn/gccl_net.h"
#include "core.h"
#include "gccl.h"

namespace gccl {

typedef char gcclNetHandle_t[GCCL_NET_HANDLE_MAXSIZE];

/* Socket Interface Selection type */
typedef enum { findSubnetIf = -1, dontCareIf = -2 } gcclSocketIfSl_t;

// Translation to external API
static const char* gcclNetName() { return gcclNet->name; }
static gcclResult_t gcclNetDevices(int* ndev, int** scores) {
  GCCLCHECK(gcclNet->devices(ndev, scores));
  return gcclSuccess;
}
static gcclResult_t gcclNetPtrSupport(int dev, int* supportedTypes) {
  GCCLCHECK(gcclNet->ptrSupport(dev, supportedTypes));
  return gcclSuccess;
}
static gcclResult_t gcclNetListen(int dev, void* handle, void** listenComm) {
  GCCLCHECK(gcclNet->listen(dev, handle, listenComm));
  return gcclSuccess;
}
static gcclResult_t gcclNetConnect(int dev, void* handle, void** sendComm) {
  GCCLCHECK(gcclNet->connect(dev, handle, sendComm));
  return gcclSuccess;
}
static gcclResult_t gcclNetAccept(void* listenComm, void** recvComm) {
  GCCLCHECK(gcclNet->accept(listenComm, recvComm));
  return gcclSuccess;
}
static gcclResult_t gcclNetIsend(void* sendComm, void* data, int size, int type,
                                 void** request) {
  GCCLCHECK(gcclNet->isend(sendComm, data, size, type, request));
  return gcclSuccess;
}
static gcclResult_t gcclNetIrecv(void* recvComm, void* data, int size, int type,
                                 void** request) {
  GCCLCHECK(gcclNet->irecv(recvComm, data, size, type, request));
  return gcclSuccess;
}
static gcclResult_t gcclNetFlush(void* recvComm, void* data, int size) {
  GCCLCHECK(gcclNet->flush(recvComm, data, size));
  return gcclSuccess;
}
static gcclResult_t gcclNetTest(void* request, int* done, int* size) {
  GCCLCHECK(gcclNet->test(request, done, size));
  return gcclSuccess;
}
static gcclResult_t gcclNetCloseSend(void* sendComm) {
  GCCLCHECK(gcclNet->closeSend(sendComm));
  return gcclSuccess;
}
static gcclResult_t gcclNetCloseRecv(void* recvComm) {
  GCCLCHECK(gcclNet->closeRecv(recvComm));
  return gcclSuccess;
}
static gcclResult_t gcclNetCloseListen(void* listenComm) {
  GCCLCHECK(gcclNet->closeListen(listenComm));
  return gcclSuccess;
}

extern bool gcclIbSupport();
extern gcclResult_t gcclSocketCreateHandle(void* opaqueHandle, const char* str);
extern gcclNet_t gcclNetIb;
// extern gcclNet_t gcclNetSocket;

}  // namespace gccl