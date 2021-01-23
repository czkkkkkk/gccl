#include "gpu/kernel/greedy_graph_allgather.h"

#include "comm/pattern/greedy_comm_pattern.h"
#include "core.h"
#include "gpu/common.h"
#include "gpu/kernel/primitives.h"

namespace gccl {

__device__ bool HasSendOrRecv(int peer_id, int n_stages, int n_peers,
                              int *send_off, int *recv_off) {
  bool ret = false;
  for (int i = 0; i < n_stages; ++i) {
    int pos = i * n_peers + peer_id;
    if (send_off[pos + 1] > send_off[pos]) {
      ret = true;
    }
    if (recv_off[pos + 1] > recv_off[pos]) {
      ret = true;
    }
  }
  return ret;
}

__global__ void GraphGreedyAllgatherKernel(CollectiveArgs args) {
  void *input = args.input;
  int tid = threadIdx.x;
  int n_threads = blockDim.x;
  int bid = blockIdx.x;
  GreedyCommPatternInfo *info =
      (GreedyCommPatternInfo *)args.info->allgather_scheme
          .comm_pattern_infos[bid]
          .data;
  int id_pos = tid / args.threads_per_conn;
  int peer_id = info->conn_peers[id_pos];
  //int peer_id = tid / WARP_SIZE;
  //if (peer_id >= args.rank) {
  //  peer_id++;
  //}
  //// Check has send size or recv size
  //if (!HasSendOrRecv(peer_id, info->n_stages, info->n_peers, info->send_off,
  //                   info->recv_off)) {
  //  return;
  //}
  int ele_size = args.ele_size;
  int feat_size = args.feat_size;
  int record_size = ele_size * feat_size;
  int threads_per_conn = args.threads_per_conn;
  // here assume record_size is a multiple of 128bits
  CopyArgs copy_args(tid % threads_per_conn, threads_per_conn,
                     info->conn_info[peer_id].my_substage_ready,
                     info->conn_info[peer_id].my_substage_done,
                     info->conn_info[peer_id].next_substage_ready,
                     info->conn_info[peer_id].prev_substage_done);
  copy_args.input = (Pack128 *)input;
  //copy_args.recv_buff = (Pack128 *)info->recv_dev_mem[peer_id]->buff;
  copy_args.recv_buff = (Pack128 *)info->conn_info[peer_id].my_recv_buff;
  copy_args.next_recv_buff = (Pack128 *)info->conn_info[peer_id].next_recv_buff;
  copy_args.extra_buff = (Pack128 *)info->dev_extra_mem;
  copy_args.extra_buff_size = info->extra_buffer_size;

  copy_args.n_128b = record_size / PACK_SIZE;
  copy_args.buff_n_128b = args.buffer_size / PACK_SIZE;
  copy_args.sync_warp = 0;
  for (int i = 0; i < info->n_stages; ++i) {
#ifdef GCCL_DEBUG
    if(tid % args.threads_per_conn == 0) {
      printf("--- Stage %d of peer %d starts\n", i, peer_id);
    }
#endif
    int pos = i * info->n_peers + peer_id;
    int send_pos = info->send_off[pos];
    int recv_pos = info->recv_off[pos];
    copy_args.send_ids = info->send_ids + send_pos;
    copy_args.recv_ids = info->recv_ids + recv_pos;
    copy_args.send_size = info->send_off[pos + 1] - info->send_off[pos];
    copy_args.recv_size = info->recv_off[pos + 1] - info->recv_off[pos];
    copy_args.max_comm_size = info->max_comm_size[i];
    Copy128b(&copy_args);
#ifdef GCCL_DEBUG
    if(tid % args.threads_per_conn == 0) {
      printf("--- Stage %d of peer %d finished\n", i, peer_id);
    }
#endif
  }
}

__global__ void GraphGreedyAllgatherBackwardKernel(CollectiveArgs args) {
  void *input = args.input;
  int tid = threadIdx.x;
  int n_threads = blockDim.x;
  int bid = blockIdx.x;
  GreedyCommPatternInfo *info =
      (GreedyCommPatternInfo *)args.info->allgather_scheme
          .comm_pattern_infos[bid]
          .data;
  int id_pos = tid / args.threads_per_conn;
  int peer_id = info->conn_peers[id_pos];
  //int peer_id = tid / WARP_SIZE;
  //if (peer_id >= args.rank) {
  //  peer_id++;
  //}
  //// Check has send size or recv size
  //if (!HasSendOrRecv(peer_id, info->n_stages, info->n_peers, info->send_off,
  //                   info->recv_off)) {
  //  return;
  //}
  int ele_size = args.ele_size;
  int feat_size = args.feat_size;
  int record_size = ele_size * feat_size;
  int threads_per_conn = args.threads_per_conn;
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

  copy_args.extra_buff = (Pack128 *)info->dev_extra_mem;
  copy_args.extra_buff_size = info->extra_buffer_size;
  copy_args.sync_warp = 0;

  for (int i = info->n_stages - 1; i >= 0; --i) {
#ifdef GCCL_DEBUG
    if(tid % info->threads_per_conn == 0) {
      printf("--- Stage %d of peer %d starts\n", i, peer_id);
    }
#endif
    int pos = i * info->n_peers + peer_id;
    int send_pos = info->send_off[pos];
    int recv_pos = info->recv_off[pos];
    copy_args.recv_ids = info->send_ids + send_pos;
    copy_args.send_ids = info->recv_ids + recv_pos;
    copy_args.recv_size = info->send_off[pos + 1] - info->send_off[pos];
    copy_args.send_size = info->recv_off[pos + 1] - info->recv_off[pos];
    copy_args.max_comm_size = info->max_comm_size[i];
    copy_args.atomic_reduce = 1;
    CopyAndReduce128b(&copy_args);
#ifdef GCCL_DEBUG
    if(tid % args.threads_per_conn == 0) {
      printf("--- Stage %d of peer %d finished\n", i, peer_id);
    }
#endif
  }
}
}  // namespace gccl
