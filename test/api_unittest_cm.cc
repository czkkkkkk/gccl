#include "gccl.h"

#include <chrono>
#include <cstdlib>
#include <random>

#include "comm/comm_info.h"
#include "comm/pattern/ring_comm_pattern.h"
#include "communicator.h"
#include "glog/logging.h"
#include "gpu/common.h"
#include "gtest/gtest.h"
#include "mpi.h"
#include "test_cuda_utils.h"
#include "test_mpi_utils.h"
#include "test_utils.h"
#include "utils.h"

namespace gccl {
namespace {

class TestAPI : public testing::Test {
 public:
  static void SetUpTestCase() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (rank == 0) {
      id = GetUniqueId();
    }
    MPI_Bcast((void*)&id, sizeof(gcclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    CommInitRank(&comm, world_size, id, rank);
  }

 protected:
  static int rank, world_size;
  static gcclUniqueId id;
  static gcclComm_t comm;
  int my_n_nodes;
  int *my_xadj, *my_adjncy;
};

int TestAPI::rank, TestAPI::world_size;
gcclUniqueId TestAPI::id;
gcclComm_t TestAPI::comm;

std::vector<int> RandVec(int n, int range) {
  std::mt19937 gen(0);
  std::uniform_int_distribution<> dist(0, n - 1);

  std::vector<int> ret(n);
  for (int i = 0; i < n; ++i) {
    ret[i] = dist(gen);
  }
  return ret;
}

// if fetch remote, fetch the remote node feature. Otherwise, set remote node
// feature to -1
std::vector<int> FetchInput(const std::vector<int>& input, int n,
                            const std::vector<int>& xadj,
                            const std::vector<int>& adjncy, int rank,
                            int world_size, int feat_size, bool fetch_remote) {
  std::vector<int> ret, extra_nodes;
  for (int i = rank; i < n; i += world_size) {
    for (int j = 0; j < feat_size; ++j) {
      ret.push_back(input[i * feat_size + j]);
    }
  }
  for (int i = 0; i < n; ++i) {
    if (i % world_size == rank) continue;
    for (int j = xadj[i]; j < xadj[i + 1]; ++j) {
      int v = adjncy[j];
      if (v % world_size == rank) {
        extra_nodes.push_back(i);
        break;
      }
    }
  }
  for (auto v : extra_nodes) {
    for (int j = 0; j < feat_size; ++j) {
      if (fetch_remote) {
        ret.push_back(input[v * feat_size + j]);
      } else
        ret.push_back(-1);
    }
  }
  return ret;
}

TEST_F(TestAPI, GraphAllgatherAllToAll) {
  if (world_size != 3) return;
  int n = 7;
  std::vector<int> xadj({0, 2, 4, 6, 6, 6, 6, 6});
  std::vector<int> adjncy({1, 2, 3, 4, 5, 6});

  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  SetConfig(comm, "{\"comm_pattern\": \"ALLTOALL\"}");
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  int feat_size = 4;
  int* input;
  std::vector<int> cpu_input, exp_output;
  if (rank == 0) {
    cpu_input = {0, 3, 6, -1, -1};
    exp_output = {0, 3, 6, 1, 2};
  } else if (rank == 1) {
    cpu_input = {1, 4, -1};
    exp_output = {1, 4, 0};
  } else {
    cpu_input = {2, 5, -1};
    exp_output = {2, 5, 0};
  }
  cpu_input = Repeat(cpu_input, feat_size);
  exp_output = Repeat(exp_output, feat_size);
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, cpu_input);
  GraphAllgather(comm, info, input, gcclInt, feat_size, 0);
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);
  SetConfig(comm, "{\"comm_pattern\": \"RING\"}");
}

TEST_F(TestAPI, GraphAllgatherAllToAllLarge) {
  int n = 50000, m = 200000;
  int feat_size = 128;
  std::vector<int> xadj, adjncy;
  std::tie(xadj, adjncy) = RandGraph(n, m);
  auto all_input = RandVec(n * feat_size, 100);
  auto my_input = FetchInput(all_input, n, xadj, adjncy, rank, world_size,
                             feat_size, false);
  auto exp_output =
      FetchInput(all_input, n, xadj, adjncy, rank, world_size, feat_size, true);
  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  SetConfig(comm, "{\"comm_pattern\": \"ALLTOALL\"}");
  int* input;
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, my_input);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  GraphAllgather(comm, info, input, gcclInt, feat_size, 0);
  cudaStreamSynchronize(0);
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);
  SetConfig(comm, "{\"comm_pattern\": \"RING\"}");
}

TEST_F(TestAPI, GraphAllgatherGreedy) {
  if (world_size != 3) return;
  int n = 7;
  std::vector<int> xadj({0, 2, 4, 6, 6, 6, 6, 6});
  std::vector<int> adjncy({1, 2, 3, 4, 5, 6});

  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  SetConfig(comm, "{\"comm_pattern\": \"GREEDY\"}");
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  int feat_size = 4;
  int* input;
  std::vector<int> cpu_input, exp_output;
  if (rank == 0) {
    cpu_input = {0, 3, 6, -1, -1};
    exp_output = {0, 3, 6, 1, 2};
  } else if (rank == 1) {
    cpu_input = {1, 4, -1};
    exp_output = {1, 4, 0};
  } else {
    cpu_input = {2, 5, -1};
    exp_output = {2, 5, 0};
  }
  cpu_input = Repeat(cpu_input, feat_size);
  exp_output = Repeat(exp_output, feat_size);
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, cpu_input);
  GraphAllgather(comm, info, input, gcclInt, feat_size, 0);
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);
  SetConfig(comm, "{\"comm_pattern\": \"RING\"}");
}

TEST_F(TestAPI, GraphAllgatherGreedyLarge) {
  int n = 50000, m = 200000;
  int feat_size = 4;
  std::vector<int> xadj, adjncy;
  std::tie(xadj, adjncy) = RandGraph(n, m);
  auto all_input = RandVec(n * feat_size, 100);
  auto my_input = FetchInput(all_input, n, xadj, adjncy, rank, world_size,
                             feat_size, false);
  auto exp_output =
      FetchInput(all_input, n, xadj, adjncy, rank, world_size, feat_size, true);
  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  SetConfig(comm, "{\"comm_pattern\": \"GREEDY\"}");
  int* input;
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, my_input);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  GraphAllgather(comm, info, input, gcclInt, feat_size, 0);
  cudaStreamSynchronize(0);
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);
  SetConfig(comm, "{\"comm_pattern\": \"RING\"}");
}

}  // namespace
}  // namespace gccl