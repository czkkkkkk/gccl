#pragma once

#include <cuda_runtime.h>

#include "core.h"

#define DIVUP(x, y) (((x) + (y)-1) / (y))
#define PACK_SIZE 16

namespace gccl {

#define WARP_SIZE 32

typedef ulong2 Pack128;
typedef uint64_t PackType;

struct Iterator2d {
  int p, v;
};

#define SetIter(iter, x, y) \
  iter.p = x;               \
  iter.v = y;

#define IncIter(iter, inc, record_size) \
  iter.p += inc.p;                      \
  iter.v += inc.v;                      \
  if (iter.v >= record_size) {          \
    iter.v -= record_size;              \
    iter.p++;                           \
  }

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

inline __device__ void Fetch128(Pack128 &v, Pack128 *p) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];"
               : "=l"(v.x), "=l"(v.y)
               : "l"(p)
               : "memory");
}
inline __device__ void Store128(Pack128 *p, Pack128 &v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(p), "l"(v.x),
               "l"(v.y)
               : "memory");
}

template <typename T>
struct MULTI {
  static_assert(sizeof(PackType) == 2 * sizeof(int32_t),
                "PackType must be twice the size of int.");
  union converter {
    PackType storage;
    struct {
      T a, b;
    };
  };

  __device__ PackType operator()(const PackType x, const PackType y) const {
    converter cx, cy, cr;
    cx.storage = x;
    cy.storage = y;
    cr.a = cx.a + cy.a;
    cr.b = cx.b + cy.b;

    return cr.storage;
  }
  __device__ void AtomicAdd(PackType *x, const PackType y) const {
    converter *cx = (converter *)x;
    converter cy;
    cy.storage = y;
    T olda = atomicAdd(&cx->a, cy.a);
    T oldb = atomicAdd(&cx->b, cy.b);
  }
  __device__ void GetVal(PackType x, T* a, T* b) const {
    converter cx;
    cx.storage = x;
    *a = cx.a;
    *b = cx.b;
  }
};

template <typename T>
struct MULTI128 {
  __device__ void operator()(Pack128 &x, Pack128 &y) {
    x.x = MULTI<T>()(x.x, y.x);
    x.y = MULTI<T>()(x.y, y.y);
  }
};

__inline__ __device__ void Print128Float(Pack128& x, const char* prefix) {
  using T = float;
  T a, b, c, d;
  MULTI<T>().GetVal(x.x, &a, &b);
  MULTI<T>().GetVal(x.y, &c, &d);
  printf("%s %f %f %f %f\n", prefix, a, b, c, d);
}

template <typename T>
inline __device__ void AtomicAdd128(Pack128 *p, Pack128 &v) {
  MULTI<T>().AtomicAdd(&p->x, v.x);
  MULTI<T>().AtomicAdd(&p->y, v.y);
}

class WaitFlag {
  volatile uint64_t *const flag;

 public:
  __host__ __device__ __forceinline__ WaitFlag(volatile uint64_t *const flag)
      : flag(flag) {}
  __device__ uint64_t get_flag() { return *flag; }
  __device__ __forceinline__ void wait(uint64_t val) {
    while ((*flag) != val) /*SPIN*/
      ;
  }
  __device__ __forceinline__ void post(uint64_t val) { *flag = val; }
  const static uint64_t FLAG_START = ~0ull >> 1;
  const static uint64_t FLAG_END = FLAG_START + 1;
};

struct CopyArgs {
  __host__ __device__ __forceinline__ CopyArgs(int tid, int n_threads,
                                               uint64_t *ready, uint64_t *done,
                                               uint64_t *next_ready,
                                               uint64_t *prev_done)
      : tid(tid),
        n_threads(n_threads),
        ready(ready),
        done(done),
        next_ready(next_ready),
        prev_done(prev_done),
        atomic_reduce(0),
        sync_warp(0) {};
  int tid, n_threads;
  Pack128 *input, *extra_buff;
  Pack128 *recv_buff, *next_recv_buff;
  const int *send_ids, *recv_ids;
  int send_size, recv_size;
  int n_128b,
      buff_n_128b;  // Per element number of 128b and buffer number of 128b
  int extra_buff_size;
  int max_comm_size;
  int atomic_reduce;
  int sync_warp;
  WaitFlag ready, done, next_ready, prev_done;
};

// Send data to next and recv data from prev at the same time
// Set to global just for test
__device__ void Copy128b(CopyArgs *args);
__device__ void CopyAndReduce128b(CopyArgs *args);

}  // namespace gccl
