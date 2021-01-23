#include "gccl.h"

#include <cuda_runtime.h>
#include <cstdio>

#include "glog/logging.h"

#include "comm/pattern/all_to_all_comm_pattern.h"
#include "comm/pattern/ring_comm_pattern.h"
#include "communicator.h"
#include "core.h"
#include "gpu/common.h"
#include "gpu/kernel/greedy_graph_allgather.h"
#include "gpu/kernel/primitives.h"
#include "param.h"

namespace gccl {

__global__ void GraphRingAllgatherKernel(CollectiveArgs args) {
  void *input = args.input;
  int tid = threadIdx.x;
  int n_threads = blockDim.x;
  int bid = blockIdx.x;
  RingCommPatternInfo *info =
      (RingCommPatternInfo *)args.info->allgather_scheme.comm_pattern_infos[bid]
          .data;
  int n_stages = info->n_stages;
  int ele_size = args.ele_size;
  int feat_size = args.feat_size;
  int record_size = ele_size * feat_size;
  // here assume record_size is a multiple of 128bits
  WaitFlag this_ready(info->forward_conn.conn_info.my_stage_ready);
  WaitFlag next_ready(info->forward_conn.conn_info.next_stage_ready);
  CopyArgs copy_args(tid, n_threads,
                     info->forward_conn.conn_info.my_substage_ready,
                     info->forward_conn.conn_info.my_substage_done,
                     info->forward_conn.conn_info.next_substage_ready,
                     info->forward_conn.conn_info.prev_substage_done);
  copy_args.input = (Pack128 *)input;
  copy_args.extra_buff = (Pack128 *)info->dev_extra_mem;
  copy_args.recv_buff = (Pack128 *)info->forward_conn.conn_info.my_recv_buff;
  copy_args.next_recv_buff =
      (Pack128 *)info->forward_conn.conn_info.next_recv_buff;

  copy_args.n_128b = record_size / PACK_SIZE;
  copy_args.buff_n_128b = args.buffer_size / PACK_SIZE;

  copy_args.extra_buff_size = info->extra_buff_size;

  copy_args.recv_ids = info->recv_ids;
  copy_args.send_ids = info->send_ids;

  int *recv_off = info->recv_off;
  int *send_off = info->send_off;

  for (int stage = 0; stage < n_stages; ++stage) {
#ifdef GCCL_DEBUG
    if (tid == 0) {
      printf("----Stage %d\n", stage);
    }
#endif
    copy_args.recv_size = recv_off[stage + 1] - recv_off[stage];
    copy_args.send_size = send_off[stage + 1] - send_off[stage];
    copy_args.max_comm_size = info->max_comm_size[stage];

    Copy128b(&copy_args);
    copy_args.send_ids += copy_args.send_size;
    copy_args.recv_ids += copy_args.recv_size;
  }
}

__global__ void GraphRingAllgatherBackwardKernel(CollectiveArgs args) {
  void *input = args.input;
  int tid = threadIdx.x;
  int n_threads = blockDim.x;
  int bid = blockIdx.x;
  RingCommPatternInfo *info =
      (RingCommPatternInfo *)args.info->allgather_scheme.comm_pattern_infos[bid]
          .data;
  int n_stages = info->n_stages;
  int ele_size = args.ele_size;
  int feat_size = args.feat_size;
  int record_size = ele_size * feat_size;
  // here assume record_size is a multiple of 128bits
  WaitFlag this_ready(info->backward_conn.conn_info.my_stage_ready);
  WaitFlag next_ready(info->backward_conn.conn_info.next_stage_ready);
  CopyArgs copy_args(tid, n_threads,
                     info->backward_conn.conn_info.my_substage_ready,
                     info->backward_conn.conn_info.my_substage_done,
                     info->backward_conn.conn_info.next_substage_ready,
                     info->backward_conn.conn_info.prev_substage_done);
  copy_args.input = (Pack128 *)input;
  copy_args.extra_buff = (Pack128 *)info->dev_extra_mem;
  copy_args.recv_buff = (Pack128 *)info->backward_conn.conn_info.my_recv_buff;
  copy_args.next_recv_buff =
      (Pack128 *)info->backward_conn.conn_info.next_recv_buff;

  copy_args.n_128b = record_size / PACK_SIZE;
  copy_args.buff_n_128b = args.buffer_size / PACK_SIZE;

  copy_args.extra_buff_size = info->extra_buff_size;

  copy_args.recv_ids = info->recv_ids;
  copy_args.send_ids = info->send_ids;

  int *recv_off = info->recv_off;
  int *send_off = info->send_off;

  for (int stage = n_stages - 1; stage >= 0; --stage) {
#ifdef GCCL_DEBUG
    if (tid == 0) {
      printf("----Stage %d\n", stage);
    }
#endif
    // Exchange send and recv ids
    copy_args.send_ids = info->recv_ids + recv_off[stage];
    copy_args.recv_ids = info->send_ids + send_off[stage];
    copy_args.send_size = recv_off[stage + 1] - recv_off[stage];
    copy_args.recv_size = send_off[stage + 1] - send_off[stage];
    copy_args.max_comm_size = info->max_comm_size[stage];

    CopyAndReduce128b(&copy_args);
  }
}

__global__ void GraphAllToAllAllgatherKernel(CollectiveArgs args) {
  void *input = args.input;
  int tid = threadIdx.x;
  int n_threads = blockDim.x;
  int bid = blockIdx.x;
  AllToAllCommPatternInfo *info =
      (AllToAllCommPatternInfo *)args.info->allgather_scheme
          .comm_pattern_infos[bid]
          .data;
  int threads_per_conn = info->threads_per_conn;
  int peer_id = tid / threads_per_conn;
  if (peer_id >= args.rank) {
    peer_id++;
  }
  int ele_size = args.ele_size;
  int feat_size = args.feat_size;
  int record_size = ele_size * feat_size;
  // here assume record_size is a multiple of 128bits
  CopyArgs copy_args(tid % threads_per_conn, threads_per_conn,
                     info->conn_info[peer_id].my_substage_ready,
                     info->conn_info[peer_id].my_substage_done,
                     info->conn_info[peer_id].next_substage_ready,
                     info->conn_info[peer_id].prev_substage_done);
  copy_args.input = (Pack128 *)input;
  copy_args.recv_buff = (Pack128 *)info->conn_info[peer_id].my_recv_buff;
  copy_args.next_recv_buff = (Pack128 *)info->conn_info[peer_id].next_recv_buff;

  copy_args.n_128b = record_size / PACK_SIZE;
  copy_args.buff_n_128b = args.buffer_size / PACK_SIZE;

  int recv_off = info->recv_off[peer_id];
  int send_off = info->send_off[peer_id];

  copy_args.recv_ids = info->recv_ids + recv_off;
  copy_args.send_ids = info->send_ids + send_off;

  copy_args.recv_size = info->recv_off[peer_id + 1] - info->recv_off[peer_id];
  copy_args.send_size = info->send_off[peer_id + 1] - info->send_off[peer_id];
  copy_args.max_comm_size = info->max_comm_size;
  Copy128b(&copy_args);
}

__global__ void GraphAllToAllAllgatherBackwardKernel(CollectiveArgs args) {
  void *input = args.input;
  int tid = threadIdx.x;
  int n_threads = blockDim.x;
  int bid = blockIdx.x;
  AllToAllCommPatternInfo *info =
      (AllToAllCommPatternInfo *)args.info->allgather_scheme
          .comm_pattern_infos[bid]
          .data;
  int threads_per_conn = info->threads_per_conn;
  int peer_id = tid / threads_per_conn;
  if (peer_id >= args.rank) {
    peer_id++;
  }
  int ele_size = args.ele_size;
  int feat_size = args.feat_size;
  int record_size = ele_size * feat_size;
  // here assume record_size is a multiple of 128bits
  CopyArgs copy_args(tid % threads_per_conn, threads_per_conn,
                     info->conn_info[peer_id].my_substage_ready,
                     info->conn_info[peer_id].my_substage_done,
                     info->conn_info[peer_id].next_substage_ready,
                     info->conn_info[peer_id].prev_substage_done);
  copy_args.input = (Pack128 *)input;
  // copy_args.recv_buff = (Pack128 *)info->recv_dev_mem[peer_id]->buff;
  copy_args.recv_buff = (Pack128 *)info->conn_info[peer_id].my_recv_buff;
  copy_args.next_recv_buff = (Pack128 *)info->conn_info[peer_id].next_recv_buff;

  copy_args.n_128b = record_size / PACK_SIZE;
  copy_args.buff_n_128b = args.buffer_size / PACK_SIZE;

  int recv_off = info->recv_off[peer_id];
  int send_off = info->send_off[peer_id];

  copy_args.recv_ids = info->send_ids + send_off;
  copy_args.send_ids = info->recv_ids + recv_off;

  copy_args.recv_size = info->send_off[peer_id + 1] - info->send_off[peer_id];
  copy_args.send_size = info->recv_off[peer_id + 1] - info->recv_off[peer_id];
  copy_args.max_comm_size = info->max_comm_size;
  copy_args.atomic_reduce = 1;
  CopyAndReduce128b(&copy_args);
}

void StartProxy(gcclComm_t comm, gcclCommInfo_t info) {
  auto &scheme = info->allgather_scheme;
  const auto &patterns = comm->GetCommScheduler()->GetCommPatterns();
  int n_blocks = scheme.n_blocks;
  for (int i = 0; i < n_blocks; ++i) {
    patterns[i]->StartProxy(comm->GetCoordinator(),
                            &scheme.comm_pattern_infos[i]);
  }
}

void SaveProxy(gcclComm_t comm, gcclCommInfo_t info, int feat_size,
               int n_threads, bool forward) {
  Config *config = comm->GetConfig();
  auto &scheme = info->allgather_scheme;
  const auto &patterns = comm->GetCommScheduler()->GetCommPatterns();
  int n_blocks = scheme.n_blocks;
  for (int i = 0; i < n_blocks; ++i) {
    patterns[i]->SaveProxy(comm->GetCoordinator(),
                           &scheme.comm_pattern_infos[i], feat_size, n_threads,
                           forward);
  }
}

void GraphAllgather(gcclComm_t comm, gcclCommInfo_t info, void *input,
                    gcclDataType_t type, int feat_size, cudaStream_t stream) {
  if (comm->GetCoordinator()->GetNPeers() == 1) {
    return;
  }
  // Build Args
  // Call Kernel
  Config *config = comm->GetConfig();
  CollectiveArgs args;
  args.ele_size = GetDataTypeSize(type);
  args.feat_size = feat_size;
  CHECK_LE(feat_size, config->max_feat_size);
  args.info = info->dev_info;
  args.input = input;
  args.rank = comm->GetCoordinator()->GetRank();
  args.buffer_size = config->buffer_size;
  int n_threads = config->n_threads;
  if (config->comm_pattern == "ALLTOALL") {
    n_threads =
        config->threads_per_conn * (comm->GetCoordinator()->GetNPeers() - 1);
  } else if (config->comm_pattern == "GREEDY") {
    n_threads = 1024 / (info->n_conn_peers * 64) * (info->n_conn_peers * 64);
    args.threads_per_conn = n_threads / info->n_conn_peers;
  }
  dim3 grid_dim(info->allgather_scheme.n_blocks);
  dim3 block_dim(n_threads);
  void *kargs[] = {&args};

  // Save proxy
  SaveProxy(comm, info, feat_size, n_threads, true);

  // End of save proxy
  if (config->comm_pattern == "RING") {
    CHECK_LE(feat_size * args.ele_size * config->n_threads, args.buffer_size);
    cudaError_t e = cudaLaunchKernel((void *)GraphRingAllgatherKernel, grid_dim,
                                     block_dim, kargs, 0, stream);
    CUDACHECKERR(e);
  } else if (config->comm_pattern == "ALLTOALL") {
    // n_threads in config does not work
    CHECK_LE(feat_size * args.ele_size * config->threads_per_conn,
             args.buffer_size);
    cudaError_t e = cudaLaunchKernel((void *)GraphAllToAllAllgatherKernel,
                                     grid_dim, block_dim, kargs, 0, stream);
    CUDACHECKERR(e);
  } else if (config->comm_pattern == "GREEDY") {
    CHECK_LE(feat_size * args.ele_size * config->threads_per_conn,
             args.buffer_size);
    cudaError_t e = cudaLaunchKernel((void *)GraphGreedyAllgatherKernel,
                                     grid_dim, block_dim, kargs, 0, stream);
    CUDACHECKERR(e);
  } else {
    CHECK(false);
  }
  // Start proxy here
  StartProxy(comm, info);

  cudaStreamSynchronize(stream);
  CUDACHECK(cudaGetLastError());
}

// float32
void GraphAllgatherBackward(gcclComm_t comm, gcclCommInfo_t info, void *input,
                            gcclDataType_t type, int feat_size,
                            cudaStream_t stream) {
  if (comm->GetCoordinator()->GetNPeers() == 1) {
    return;
  }
  Config *config = comm->GetConfig();
  CollectiveArgs args;
  CHECK(type == gccl::gcclDataType_t::gcclFloat);

  args.ele_size = GetDataTypeSize(type);

  args.feat_size = feat_size;
  CHECK_LE(feat_size, config->max_feat_size);
  args.info = info->dev_info;
  args.input = input;
  args.rank = comm->GetCoordinator()->GetRank();
  args.buffer_size = config->buffer_size;
  int n_threads = config->n_threads;
  if (config->comm_pattern == "ALLTOALL") {
    n_threads =
        config->threads_per_conn * (comm->GetCoordinator()->GetNPeers() - 1);
  } else if (config->comm_pattern == "GREEDY") {
    n_threads = 1024 / (info->n_conn_peers * 64) * (info->n_conn_peers * 64);
    args.threads_per_conn = n_threads / info->n_conn_peers;
  }
  dim3 grid_dim(info->allgather_scheme.n_blocks);
  dim3 block_dim(n_threads);
  void *kargs[] = {&args};

  // Save proxy
  SaveProxy(comm, info, feat_size, n_threads, false);

  // End of save proxy

  if (config->comm_pattern == "RING") {
    cudaError_t e = cudaLaunchKernel((void *)GraphRingAllgatherBackwardKernel,
                                     grid_dim, block_dim, kargs, 0, stream);
    CUDACHECKERR(e);
  } else if (config->comm_pattern == "ALLTOALL") {
    // n_threads in config does not work
    cudaError_t e =
        cudaLaunchKernel((void *)GraphAllToAllAllgatherBackwardKernel, grid_dim,
                         block_dim, kargs, 0, stream);
    CUDACHECKERR(e);
  } else if (config->comm_pattern == "GREEDY") {
    // n_threads in config does not work
    int threads_per_conn = config->threads_per_conn;
    cudaError_t e = cudaLaunchKernel((void *)GraphGreedyAllgatherBackwardKernel,
                                     grid_dim, block_dim, kargs, 0, stream);
    CUDACHECKERR(e);
  } else {
    CHECK(false);
  }
  // Start proxy here
  StartProxy(comm, info);
  cudaStreamSynchronize(stream);
  CUDACHECK(cudaGetLastError());
}
}  // namespace gccl
