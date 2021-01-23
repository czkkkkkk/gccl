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

template <typename T>
std::vector<T> RandVec(int n, int range) {
  std::mt19937 gen(0);
  std::uniform_int_distribution<> dist(0, n - 1);

  std::vector<T> ret(n);
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

std::vector<float> FetchGradient(const std::vector<float>& input, int n,
                            const std::vector<int>& xadj,
                            const std::vector<int>& adjncy, int rank,
                            int world_size, int feat_size, bool fetch_remote) {
  std::vector<float> ret;
  std::vector<int> extra_nodes;
  for (int i = rank; i < n; i += world_size) {
    for (int j = 0; j < feat_size; ++j) {
      ret.push_back(input[i * feat_size + j]);
    }
  }
  if(fetch_remote) {

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
        ret.push_back(input[v * feat_size + j]);
      }
    }
  } else {
    for(int i = rank; i < n; i += world_size) {
      std::set<int> nei_set;
      nei_set.insert(rank);
      for (int j = xadj[i]; j < xadj[i + 1]; ++j) {
        int v = adjncy[j];
        if(v % world_size == rank) continue;
        nei_set.insert(v % world_size);
      }
      for(int j = 0; j < feat_size; ++j) {
        ret[(i / 3) * feat_size + j] *= nei_set.size();
      }
    }
  }
  return ret;
}


TEST_F(TestAPI, GraphPartition) {
  int n = 7;
  std::vector<int> xadj({0, 2, 4, 6, 6, 6, 6, 6});
  std::vector<int> adjncy({1, 2, 3, 4, 5, 6});

  gcclCommInfo_t info;
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
}

TEST_F(TestAPI, DispatchData) {
  if (world_size != 3) return;
  int n = 7;
  std::vector<int> xadj({0, 2, 4, 6, 6, 6, 6, 6});
  std::vector<int> adjncy({1, 2, 3, 4, 5, 6});

  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  std::vector<float> data, local_data, exp_data;
  int local_n_nodes;
  int feat_size = 2;
  if (rank == 0) {
    data = {7, 1, 2, 3, 4, 5, 6};
    local_n_nodes = 5;
    exp_data = {7, 3, 6, 1, 2};
  } else if (rank == 1) {
    local_n_nodes = 3;
    exp_data = {1, 4, 7};
  } else {
    local_n_nodes = 3;
    exp_data = {2, 5, 7};
  }
  data = Repeat(data, feat_size);
  local_data.resize(feat_size * local_n_nodes);
  exp_data = Repeat(exp_data, feat_size);
  DispatchFloat(comm, data.data(), feat_size, local_n_nodes, local_data.data(),
                0);
  EXPECT_VEC_EQ(local_data, exp_data);
}

TEST_F(TestAPI, GraphPartitionWorldSize3) {
  if (world_size != 3) return;
  int n = 7;
  std::vector<int> xadj({0, 2, 4, 6, 6, 6, 6, 6});
  std::vector<int> adjncy({1, 2, 3, 4, 5, 6});

  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  EXPECT_EQ(info->allgather_scheme.n_blocks, 1);
  struct ExpRes {
    int n_stages;
    std::vector<int> send_ids;
    std::vector<int> send_off;
    std::vector<int> recv_ids;
    std::vector<int> recv_off;
  } exp_res;
  if (rank == 0) {
    exp_res = {2, {0}, {0, 1, 1}, {4, 3}, {0, 1, 2}};
  } else if (rank == 1) {
    exp_res = {2, {0, 2}, {0, 1, 2}, {2}, {0, 1, 1}};
  } else {
    exp_res = {2, {0, -1}, {0, 1, 2}, {-1, 2}, {0, 1, 2}};
  }
  auto& pattern_info = info->allgather_scheme.comm_pattern_infos[0];
  auto* ring_info = pattern_info.GetRingCommPatternInfo();
  EXPECT_EQ(ring_info->n_stages, exp_res.n_stages);
  EXPECT_GPU_CPU_VEC_EQ(ring_info->send_ids, exp_res.send_ids);
  EXPECT_GPU_CPU_VEC_EQ(ring_info->send_off, exp_res.send_off);
  EXPECT_GPU_CPU_VEC_EQ(ring_info->recv_ids, exp_res.recv_ids);
  EXPECT_GPU_CPU_VEC_EQ(ring_info->recv_off, exp_res.recv_off);
}

TEST_F(TestAPI, GraphAllgather) {
  if (world_size != 3) return;
  int n = 7;
  std::vector<int> xadj({0, 2, 4, 6, 6, 6, 6, 6});
  std::vector<int> adjncy({1, 2, 3, 4, 5, 6});

  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  int feat_size = 128;
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
}

TEST_F(TestAPI, GraphAllgatherBackward) {
  if (world_size != 3) return;
  int n = 7;
  std::vector<int> xadj({0, 2, 4, 6, 6, 6, 6, 6});
  std::vector<int> adjncy({1, 2, 3, 4, 5, 6});

  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  int feat_size = 128;
  float* input;
  std::vector<float> cpu_input, exp_output;
  if (rank == 0) {
    cpu_input = {1, 1, 1, 1, 1};
    exp_output = {3, 1, 1, 1, 1};
  } else if (rank == 1) {
    cpu_input = {1, 1, 1};
    exp_output = {2, 1, 2};
  } else {
    cpu_input = {1, 1, 1};
    exp_output = {2, 1, 1};
  }
  cpu_input = Repeat(cpu_input, feat_size);
  exp_output = Repeat(exp_output, feat_size);
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, cpu_input);
  GraphAllgatherBackward(comm, info, input, gcclFloat, feat_size, 0);
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);
}

TEST_F(TestAPI, GraphAllgatherBackwardLarge) {
  int n = 50000, m = 100000;
  int feat_size = 64;
  std::vector<int> xadj, adjncy;
  std::tie(xadj, adjncy) = RandGraph(n, m);
  auto all_input = RandVec<float>(n * feat_size, 100);
  // auto all_input = std::vector<float>(n * feat_size, 1);
  auto my_input = FetchGradient(all_input, n, xadj, adjncy, rank, world_size,
                             feat_size, true);
  auto exp_output =
      FetchGradient(all_input, n, xadj, adjncy, rank, world_size, feat_size, false);
  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  float* input;
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, my_input);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  GraphAllgatherBackward(comm, info, input, gcclFloat, feat_size, 0);
  cudaStreamSynchronize(0);
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);
}

TEST_F(TestAPI, GraphGreedyAllgatherBackward) {
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
  float* input;
  std::vector<float> cpu_input, exp_output;
  if (rank == 0) {
    cpu_input = {1, 1, 1, 1, 1};
    exp_output = {3, 1, 1, 1, 1};
  } else if (rank == 1) {
    cpu_input = {1, 1, 1};
    exp_output = {2, 1, 1};
  } else {
    cpu_input = {1, 1, 1};
    exp_output = {2, 1, 1};
  }
  cpu_input = Repeat(cpu_input, feat_size);
  exp_output = Repeat(exp_output, feat_size);
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, cpu_input);
  GraphAllgatherBackward(comm, info, input, gcclFloat, feat_size, 0);
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);
  SetConfig(comm, "{\"comm_pattern\": \"RING\"}");
}

TEST_F(TestAPI, GraphAllgatherGreedyBackwardLarge) {
  int n = 50000, m = 100000;
  int feat_size = 64;
  std::vector<int> xadj, adjncy;
  std::tie(xadj, adjncy) = RandGraph(n, m);
  auto all_input = RandVec<float>(n * feat_size, 100);
  // auto all_input = std::vector<float>(n * feat_size, 1);
  auto my_input = FetchGradient(all_input, n, xadj, adjncy, rank, world_size,
                             feat_size, true);
  auto exp_output =
      FetchGradient(all_input, n, xadj, adjncy, rank, world_size, feat_size, false);
  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  SetConfig(comm, "{\"comm_pattern\": \"GREEDY\"}");
  float* input;
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, my_input);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  GraphAllgatherBackward(comm, info, input, gcclFloat, feat_size, 0);
  cudaStreamSynchronize(0);
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);
  SetConfig(comm, "{\"comm_pattern\": \"RING\"}");
}

TEST_F(TestAPI, GraphAllToAllAllgatherBackward) {
  if (world_size != 3) return;
  int n = 7;
  std::vector<int> xadj({0, 2, 4, 6, 6, 6, 6, 6});
  std::vector<int> adjncy({1, 2, 3, 4, 5, 6});

  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  SetConfig(comm, "{\"comm_pattern\": \"ALLTOALL\"}");
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  int feat_size = 64;
  float* input;
  std::vector<float> cpu_input, exp_output;
  if (rank == 0) {
    cpu_input = {1, 1, 1, 1, 1};
    exp_output = {3, 1, 1, 1, 1};
  } else if (rank == 1) {
    cpu_input = {1, 1, 1};
    exp_output = {2, 1, 1};
  } else {
    cpu_input = {1, 1, 1};
    exp_output = {2, 1, 1};
  }
  cpu_input = Repeat(cpu_input, feat_size);
  exp_output = Repeat(exp_output, feat_size);
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, cpu_input);
  GraphAllgatherBackward(comm, info, input, gcclFloat, feat_size, 0);
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);
  SetConfig(comm, "{\"comm_pattern\": \"RING\"}");
}

TEST_F(TestAPI, GraphAllgatherAllToAllBackwardLarge) {
  int n = 50000, m = 100000;
  int feat_size = 64;
  std::vector<int> xadj, adjncy;
  std::tie(xadj, adjncy) = RandGraph(n, m);
  auto all_input = RandVec<float>(n * feat_size, 100);
  // auto all_input = std::vector<float>(n * feat_size, 1);
  auto my_input = FetchGradient(all_input, n, xadj, adjncy, rank, world_size,
                             feat_size, true);
  auto exp_output =
      FetchGradient(all_input, n, xadj, adjncy, rank, world_size, feat_size, false);
  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  SetConfig(comm, "{\"comm_pattern\": \"ALLTOALL\"}");
  float* input;
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, my_input);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  GraphAllgatherBackward(comm, info, input, gcclFloat, feat_size, 0);
  cudaStreamSynchronize(0);
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);
  SetConfig(comm, "{\"comm_pattern\": \"RING\"}");
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
  int feat_size = 128;
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

TEST_F(TestAPI, GraphAllgatherShmConn) {
  if (world_size != 3) return;
  int n = 7;
  std::vector<int> xadj({0, 2, 4, 6, 6, 6, 6, 6});
  std::vector<int> adjncy({1, 2, 3, 4, 5, 6});

  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  setenv("GCCL_TRANSPORT_LEVEL", "shm", 1);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  int feat_size = 128;
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
  unsetenv("GCCL_TRANSPORT_LEVEL");
}


TEST_F(TestAPI, GraphAllgatherLarge) {
  int n = 50000, m = 200000;
  int feat_size = 128;
  std::vector<int> xadj, adjncy;
  std::tie(xadj, adjncy) = RandGraph(n, m);
  auto all_input = RandVec<int>(n * feat_size, 100);
  auto my_input = FetchInput(all_input, n, xadj, adjncy, rank, world_size,
                             feat_size, false);
  auto exp_output =
      FetchInput(all_input, n, xadj, adjncy, rank, world_size, feat_size, true);
  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  int* input;
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, my_input);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  GraphAllgather(comm, info, input, gcclInt, feat_size, 0);
  cudaStreamSynchronize(0);
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);
}

TEST_F(TestAPI, GraphAllgatherAllToAllLarge) {
  int n = 50000, m = 200000;
  int feat_size = 128;
  std::vector<int> xadj, adjncy;
  std::tie(xadj, adjncy) = RandGraph(n, m);
  auto all_input = RandVec<int>(n * feat_size, 100);
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

TEST_F(TestAPI, GraphAllgatherGreedyLarge) {
  int n = 50000, m = 200000;
  int feat_size = 4;
  std::vector<int> xadj, adjncy;
  std::tie(xadj, adjncy) = RandGraph(n, m);
  auto all_input = RandVec<int>(n * feat_size, 100);
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

TEST_F(TestAPI, GraphAllgatherShmConnLarge) {
  int n = 50000, m = 200000;
  int feat_size = 128;
  std::vector<int> xadj, adjncy;
  std::tie(xadj, adjncy) = RandGraph(n, m);
  auto all_input = RandVec<int>(n * feat_size, 100);
  auto my_input = FetchInput(all_input, n, xadj, adjncy, rank, world_size,
                             feat_size, false);
  auto exp_output =
      FetchInput(all_input, n, xadj, adjncy, rank, world_size, feat_size, true);
  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  setenv("GCCL_TRANSPORT_LEVEL", "shm", 1);
  int* input;
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, my_input);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  GraphAllgather(comm, info, input, gcclInt, feat_size, 0);
  cudaStreamSynchronize(0);
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);
  unsetenv("GCCL_TRANSPORT_LEVEL");
}

TEST_F(TestAPI, GraphAllgatherMultiBlocks) {
  std::vector<int> perm(world_size);
  std::iota(perm.begin(), perm.end(), 0);
  std::string str = "[[";
  str += VecToString(perm, ',');
  str += "],[";
  std::reverse(perm.begin(), perm.end());
  str += VecToString(perm, ',');
  str += "]]";
  SetConfig(comm, std::string("{\"n_blocks\": 2, \"rings\":") + str + "}");
  int n = 60001, m = 320003;
  int feat_size = 128;
  std::vector<int> xadj, adjncy;
  std::tie(xadj, adjncy) = RandGraph(n, m);
  auto all_input = RandVec<int>(n * feat_size, 100);
  auto my_input = FetchInput(all_input, n, xadj, adjncy, rank, world_size,
                             feat_size, false);
  auto exp_output =
      FetchInput(all_input, n, xadj, adjncy, rank, world_size, feat_size, true);
  gcclCommInfo_t info;
  setenv("GCCL_PART_OPT", "NAIVE", 1);
  int* input;
  GCCLSetCudaDevice(comm->GetCoordinator()->GetDevId());
  GCCLMallocAndCopy(&input, my_input);
  PartitionGraph(comm, n, xadj.data(), adjncy.data(), &info, &my_n_nodes,
                 &my_xadj, &my_adjncy);
  GraphAllgather(comm, info, input, gcclInt, feat_size, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  EXPECT_GPU_CPU_VEC_EQ(input, exp_output);

  std::reverse(perm.begin(), perm.end());
  str = "[[" + VecToString(perm, ',') + "]]";
  SetConfig(comm, std::string("{\"n_blocks\": 1, \"rings\":") + str + "}");
}

}  // namespace
}  // namespace gccl
