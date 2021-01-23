#include "gpu/kernel/primitives.h"

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

#include "comm/pattern/comm_pattern.h"
#include "gpu/common.h"

namespace gccl {

#define MAX_ID_SIZE_PER_SUBSTAGES 8192
#define UNROLL 4

__device__ void FetchIds(int *dst_ids, const int *src_ids, int size, int tid,
                         int n_threads, int sync_warp) {
  while (tid < size) {
    dst_ids[tid] = src_ids[tid];
    tid += n_threads;
  }
  // FIXME in all to all comm, not all threads need to sync
  if(sync_warp) {
    __syncwarp();
  } else {
    __syncthreads();
  }
}

__device__ void Copy128b(CopyArgs *args) {
  // Assume args->buff_n_128b >= args->n_128b * args->n_threads
  int per_substage_n_ele =
      args->buff_n_128b / (args->n_128b * args->n_threads) * args->n_threads;
  // int recv_n_substages = DIVUP(args->recv_size, per_substage_n_ele);
  // int send_n_substages = DIVUP(args->send_size, per_substage_n_ele);
  int n_substages = DIVUP(args->max_comm_size, per_substage_n_ele);
  int total_size_per_substage = per_substage_n_ele * args->n_128b;
  // int n_substages = MAX(recv_n_substages, send_n_substages);
  Iterator2d recv_iter, send_iter, recv_inc, send_inc;
  SetIter(recv_iter, args->tid / args->n_128b, args->tid % args->n_128b);
  SetIter(send_iter, args->tid / args->n_128b, args->tid % args->n_128b);
  SetIter(recv_inc, args->n_threads / args->n_128b,
          args->n_threads % args->n_128b);
  SetIter(send_inc, args->n_threads / args->n_128b,
          args->n_threads % args->n_128b);

  CUDA_DEBUG(
      printf("Send size %d, recv size %d\n", args->send_size, args->recv_size));
  CUDA_DEBUG(printf("n_substages %d\n", n_substages));
  CUDA_DEBUG(printf("extra_buff_size is %d\n", args->extra_buff_size));
  // if (args->tid == 0) {
  //  args->ready.post(WaitFlag::FLAG_START);
  //  args->next_ready.wait(WaitFlag::FLAG_START);
  //  args->done.post(WaitFlag::FLAG_START);
  //  args->prev_done.wait(WaitFlag::FLAG_START);
  //}
  //__shared__ int shared_ids[MAX_ID_SIZE_PER_SUBSTAGES];
  // int send_off = 0, recv_off = 0;
  // if(args->tid == 0) {
  //  assert(MAX_ID_SIZE_PER_SUBSTAGES >= per_substage_n_ele);
  //}

  for (int substage = 0; substage < n_substages; ++substage) {
    // Set substage ready
    // Wait next substage ready
    // Send to next
    // Set done
    // Wait prev done
    // Recv buff
    CUDA_DEBUG(printf("----------------Substage %d\n", substage));
    if (args->tid == 0) {
      CUDA_DEBUG(printf("-------Posting my ready %d\n", substage + 1));
      args->ready.post(substage + 1);
      CUDA_DEBUG(printf("-------Waiting next ready %d\n", substage + 1));
      args->next_ready.wait(substage + 1);
      CUDA_DEBUG(printf("-------Finished waiting next_ready %d \n", substage + 1));
    }
    if(args->sync_warp) {
      __syncwarp();
    } else {
      __syncthreads();
    }

    // int substage_send_size =
    //    MIN(per_substage_n_ele, args->send_size - send_off);
    // FetchIds(shared_ids, args->send_ids + send_off, substage_send_size,
    //         args->tid, args->n_threads);

    int buff_iter = args->tid;
    int buff_inc = args->n_threads;
    // While loop for sending data
    while (buff_iter < total_size_per_substage &&
           send_iter.p < args->send_size) {
      Pack128 *send_ptr;
      Pack128 t0[UNROLL];
      int org_buff_iter = buff_iter;
      int size = 0;
      /*
      for(int i = 0; i < UNROLL; ++i) {
        if(buff_iter < total_size_per_substage && send_iter.p < args->send_size)
      { int send_id = shared_ids[send_iter.p - send_off]; if (send_id < 0) {
            send_id = ENCODE(send_id);
            send_ptr = args->extra_buff + send_id * args->n_128b + send_iter.v;
          } else {
            send_ptr = args->input + send_id * args->n_128b + send_iter.v;
          }
          Fetch128(t0[i], send_ptr);
          IncIter(send_iter, send_inc, args->n_128b);
          size++;
          buff_iter += buff_inc;
        }
        else {
          break;
        }
      }
      for(int i = 0; i < size; ++i) {
        Pack128 *rem_buff_ptr = args->next_recv_buff + org_buff_iter;
        org_buff_iter += buff_inc;
        Store128(rem_buff_ptr, t0[i]);
      }
      */
      // /*
      int send_id = args->send_ids[send_iter.p];
      // int send_id = shared_ids[send_iter.p - send_off];
      if (send_id < 0) {
        send_id = ENCODE(send_id);
        send_ptr = args->extra_buff + send_id * args->n_128b + send_iter.v;
      } else {
        send_ptr = args->input + send_id * args->n_128b + send_iter.v;
      }
      Pack128 *rem_buff_ptr = args->next_recv_buff + buff_iter;
      Pack128 v;
      Fetch128(v, send_ptr);
      Store128(rem_buff_ptr, v);
      // printf("Send value %llu %llu %llu %llu\n",v.x & ((1ull <<
      // 32) - 1), v.x >> 32,  v.y & ((1ull << 32) - 1), v.y >> 32);

      buff_iter += buff_inc;
      IncIter(send_iter, send_inc, args->n_128b);
      // */
    }
    // send_off += substage_send_size;

    if (args->tid == 0) {
      CUDA_DEBUG(printf("-------Posting my done %d\n", substage + 1));
      args->done.post(substage + 1);
      CUDA_DEBUG(printf("-------Waiting prev done %d\n", substage + 1));
      args->prev_done.wait(substage + 1);
      CUDA_DEBUG(printf("-------Finished waiting prev done %d \n", substage + 1));
    }
    if(args->sync_warp) {
      __syncwarp();
    } else {
      __syncthreads();
    }
    // int substage_recv_size =
    //    MIN(per_substage_n_ele, args->recv_size - recv_off);

    // FetchIds(shared_ids, args->recv_ids + recv_off, substage_recv_size,
    //         args->tid, args->n_threads);

    buff_iter = args->tid;
    buff_inc = args->n_threads;
    while (buff_iter < total_size_per_substage &&
           recv_iter.p < args->recv_size) {
      Pack128 t0[UNROLL];
      int size = 0;
      Iterator2d org_recv_iter = recv_iter;
      Pack128 *recv_ptr;
      /*
      for(int i = 0; i < UNROLL; ++i) {
        if(buff_iter < total_size_per_substage && recv_iter.p < args->recv_size)
      { Pack128 *buff_ptr = args->recv_buff + buff_iter; Fetch128(t0[i],
      buff_ptr); size ++; IncIter(recv_iter, recv_inc, args->n_128b); buff_iter
      += buff_inc;
        }
        else {
          break;
        }
      }
      for(int i = 0; i < size; ++i) {
        int recv_id = shared_ids[org_recv_iter.p - recv_off];
        if (recv_id < 0) {
          recv_id = ENCODE(recv_id);
          recv_ptr = args->extra_buff + recv_id * args->n_128b +
      org_recv_iter.v; } else { recv_ptr = args->input + recv_id * args->n_128b
      + org_recv_iter.v;
        }
        Store128(recv_ptr, t0[i]);
        IncIter(org_recv_iter, recv_inc, args->n_128b);
      }
      */
      // /*
      Pack128 v;
      Pack128 *buff_ptr = args->recv_buff + buff_iter;

      int recv_id = args->recv_ids[recv_iter.p];
      // int recv_id = shared_ids[recv_iter.p - recv_off];

      if (recv_id < 0) {
        recv_id = ENCODE(recv_id);
        assert(recv_id < args->extra_buff_size);
        recv_ptr = args->extra_buff + recv_id * args->n_128b + recv_iter.v;
      } else {
        recv_ptr = args->input + recv_id * args->n_128b + recv_iter.v;
      }
      Fetch128(v, buff_ptr);
      Store128(recv_ptr, v);
      // printf("Fetch value %llu %llu %llu %llu\n",v.x & ((1ull <<
      // 32) - 1), v.x >> 32,  v.y & ((1ull << 32) - 1), v.y >> 32);
      buff_iter += buff_inc;
      IncIter(recv_iter, recv_inc, args->n_128b);
      // */
    }
    // recv_off += substage_recv_size;
  }
  // Reset flags
  if (args->tid == 0) {
    CUDA_DEBUG(printf("----Posting ready flag end\n"));
    args->ready.post(WaitFlag::FLAG_END);
    CUDA_DEBUG(printf("----Waiting next_ready flag end\n"));
    args->next_ready.wait(WaitFlag::FLAG_END);
    CUDA_DEBUG(printf("----Posting done flag end\n"));
    args->done.post(WaitFlag::FLAG_END);
    CUDA_DEBUG(printf("----Waiting prev done flag end\n"));
    args->prev_done.wait(WaitFlag::FLAG_END);
    CUDA_DEBUG(printf("----Finished\n"));
    
  }
    if(args->sync_warp) {
      __syncwarp();
    } else {
      __syncthreads();
    }
}

__device__ void CopyAndReduce128b(CopyArgs *args) {
  // Assume args->buff_n_128b >= args->n_128b * args->n_threads
  int per_substage_n_ele =
      args->buff_n_128b / (args->n_128b * args->n_threads) * args->n_threads;
  int n_substages = DIVUP(args->max_comm_size, per_substage_n_ele);
  int total_size_per_substage = per_substage_n_ele * args->n_128b;
  Iterator2d recv_iter, send_iter, recv_inc, send_inc;
  SetIter(recv_iter, args->tid / args->n_128b, args->tid % args->n_128b);
  SetIter(send_iter, args->tid / args->n_128b, args->tid % args->n_128b);
  SetIter(recv_inc, args->n_threads / args->n_128b,
          args->n_threads % args->n_128b);
  SetIter(send_inc, args->n_threads / args->n_128b,
          args->n_threads % args->n_128b);

  CUDA_DEBUG(printf("max comm size is %d\n", args->max_comm_size));
  CUDA_DEBUG(
      printf("Send size %d, recv size %d\n", args->send_size, args->recv_size));
  CUDA_DEBUG(printf("n_substages %d\n", n_substages));
  CUDA_DEBUG(printf("extra_buff_size is %d\n", args->extra_buff_size));

  for (int substage = 0; substage < n_substages; ++substage) {
    // Set substage ready
    // Wait next substage ready
    // Send to next
    // Set done
    // Wait prev done
    // Recv buff
    CUDA_DEBUG(printf("----------------Substage %d\n", substage));
    if (args->tid == 0) {
      args->ready.post(substage + 1);
      args->next_ready.wait(substage + 1);
    }
    if(args->sync_warp) {
      __syncwarp();
    } else {
      __syncthreads();
    }
    // int substage_send_size =
    //    MIN(per_substage_n_ele, args->send_size - send_off);
    // FetchIds(shared_ids, args->send_ids + send_off, substage_send_size,
    //         args->tid, args->n_threads);

    int buff_iter = args->tid;
    int buff_inc = args->n_threads;
    // While loop for sending data
    while (buff_iter < total_size_per_substage &&
           send_iter.p < args->send_size) {
      Pack128 *send_ptr;
      Pack128 t0[UNROLL];
      int org_buff_iter = buff_iter;
      int size = 0;
      int send_id = args->send_ids[send_iter.p];
      if (send_id < 0) {
        send_id = ENCODE(send_id);
        send_ptr = args->extra_buff + send_id * args->n_128b + send_iter.v;
      } else {
        send_ptr = args->input + send_id * args->n_128b + send_iter.v;
      }
      Pack128 *rem_buff_ptr = args->next_recv_buff + buff_iter;
      Pack128 v;
      Fetch128(v, send_ptr);
      Store128(rem_buff_ptr, v);
      // printf("Send value %llu %llu %llu %llu\n",v.x & ((1ull <<
      // 32) - 1), v.x >> 32,  v.y & ((1ull << 32) - 1), v.y >> 32);

      buff_iter += buff_inc;
      IncIter(send_iter, send_inc, args->n_128b);
    }

    if (args->tid == 0) {
      args->done.post(substage + 1);
      args->prev_done.wait(substage + 1);
    }
    if(args->sync_warp) {
      __syncwarp();
    } else {
      __syncthreads();
    }

    buff_iter = args->tid;
    buff_inc = args->n_threads;
    while (buff_iter < total_size_per_substage &&
           recv_iter.p < args->recv_size) {
      Pack128 t0[UNROLL];
      int size = 0;
      Iterator2d org_recv_iter = recv_iter;
      Pack128 *recv_ptr;
      Pack128 v;
      Pack128 *buff_ptr = args->recv_buff + buff_iter;

      int recv_id = args->recv_ids[recv_iter.p];
      // int recv_id = shared_ids[recv_iter.p - recv_off];

      bool reduce = false;
      if (recv_id < 0) {
        recv_id = ENCODE(recv_id);
        recv_ptr = args->extra_buff + recv_id * args->n_128b + recv_iter.v;
      } else {
        reduce = true;
        recv_ptr = args->input + recv_id * args->n_128b + recv_iter.v;
      }
      Fetch128(v, buff_ptr);
      if (!args->atomic_reduce) {
        if (reduce) {
          Pack128 u;
          Fetch128(u, recv_ptr);
          MULTI128<float>()(v, u);
        }
        Store128(recv_ptr, v);
      } else {
        // Atomic op must reduce
        AtomicAdd128<float>(recv_ptr, v);
      }
      // printf("Fetch value %llu %llu %llu %llu\n",v.x & ((1ull <<
      // 32) - 1), v.x >> 32,  v.y & ((1ull << 32) - 1), v.y >> 32);
      buff_iter += buff_inc;
      IncIter(recv_iter, recv_inc, args->n_128b);
      // */
    }
    // recv_off += substage_recv_size;
  }
  // Reset flags
  if (args->tid == 0) {
    args->ready.post(WaitFlag::FLAG_END);
    args->next_ready.wait(WaitFlag::FLAG_END);
    args->done.post(WaitFlag::FLAG_END);
    args->prev_done.wait(WaitFlag::FLAG_END);
  }
  if(args->sync_warp) {
    __syncwarp();
  } else {
    __syncthreads();
  }
}

}  // namespace gccl
