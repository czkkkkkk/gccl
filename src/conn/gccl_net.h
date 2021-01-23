/*************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include "gccl.h"

namespace gccl {

#define GCCL_NET_MAJOR 1
#define GCCL_NET_MINOR 0

#define GCCL_NET_HANDLE_MAXSIZE 64

#define GCCL_PTR_HOST 0x1
#define GCCL_PTR_CUDA 0x2

#define GCCL_MAX_SCORE 0x7

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Return the number of network devices along with their scores relative to
  // the current CUDA device. The per device score should be a value from 1-7
  // with a higher score representing a better choice for performance. This call
  // should allocate the 'scores' array using malloc(3), and it will then be
  // freed automatically by GCCL.
  gcclResult_t (*devices)(int* ndev, int** scores);
  // Return whether this device supports host pointers and/or CUDA pointers
  // as data from the current GPU. Supported types should be composed with
  // GCCL_PTR_HOST and NCCL_PTR_CUDA.
  gcclResult_t (*ptrSupport)(int dev, int* supportedTypes);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to GCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  gcclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  gcclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called
  // connectHandle
  gcclResult_t (*accept)(void* listenComm, void** recvComm);
  // Asynchronous send to a peer. Type is either GCCL_PTR_HOST or NCCL_PTR_CUDA.
  gcclResult_t (*isend)(void* sendComm, void* data, int size, int type,
                        void** request);
  // Asynchronous recv from a peer. Type is either GCCL_PTR_HOST or
  // GCCL_PTR_CUDA.
  gcclResult_t (*irecv)(void* recvComm, void* data, int size, int type,
                        void** request);
  // Perform a flush/fence to make sure all data received with GCCL_PTR_CUDA is
  // visible to the GPU
  gcclResult_t (*flush)(void* recvComm, void* data, int size);
  // Test whether a request is complete and return the size received (can be
  // less than requested).
  gcclResult_t (*test)(void* request, int* done, int* size);
  // Close and free send/recv comm objects
  gcclResult_t (*closeSend)(void* sendComm);
  gcclResult_t (*closeRecv)(void* recvComm);
  gcclResult_t (*closeListen)(void* listenComm);
} gcclNet_t;

extern
#ifdef __cplusplus
    "C"
#endif
    gcclNet_t* gcclNet;

}  // namespace gccl
