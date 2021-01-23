#include "gccl.h"

#include <cuda_runtime.h>
#include <chrono>
#include <ratio>
#include <tuple>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "mpi.h"

#include "comm/comm_info.h"
#include "communicator.h"
#include "coordinator.h"

DEFINE_int32(n_nodes, 100000, "Number of nodes");
DEFINE_int32(n_edges, 500000, "Number of edges");
DEFINE_int32(feat_size, 128, "Feature size");
DEFINE_string(input_graph, "", "Input graph file");
DEFINE_string(cached_graph_dir, "", "Cached graph dir");

void BcastString(std::string *str, int rank) {
  const int MAX_BYTES = 128;
  char buff[MAX_BYTES];
  int size;
  if (rank == 0) {
    memcpy(buff, str->data(), str->size());
    size = str->size();
  }
  MPI_Bcast((void *)&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast((void *)buff, MAX_BYTES, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    *str = std::string(buff, buff + size);
  }
}

using TP = std::chrono::steady_clock::time_point;

TP GetTime() { return std::chrono::steady_clock::now(); }

long int TimeDiff(const TP &begin, const TP &end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
      .count();
}

void run() {
  // Generate Random Graph
  // Build Comm Info
  // Graph allgather
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  gccl::gcclUniqueId id;
  gccl::gcclComm_t comm;
  if (rank == 0) id = gccl::GetUniqueId();
  MPI_Bcast((void *)&id, sizeof(gccl::gcclUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  CommInitRank(&comm, world_size, id, rank);

  gccl::gcclCommInfo_t info;

  int n_nodes = FLAGS_n_nodes;
  int n_edges = FLAGS_n_edges;
  int my_n_nodes, *my_xadj, *my_adjncy;
  int feat_size = FLAGS_feat_size;
  cudaSetDevice(comm->GetCoordinator()->GetDevId());
  std::vector<int> xadj, adjncy;
  std::string cached_graph_dir = FLAGS_cached_graph_dir;
  if (rank == 0) {
    std::string input_graph = FLAGS_input_graph;
    if(cached_graph_dir.size() > 0) {
      LOG(INFO) << "Using cached graph dir " << cached_graph_dir;
    } else {

      if (input_graph == "") {
        LOG(INFO) << "Generate random graph with nodes " << n_nodes << " edges "
                  << n_edges;
        std::tie(xadj, adjncy) = gccl::RandGraph(n_nodes, n_edges);
      } else {
        LOG(INFO) << "Using real graph " << input_graph;
        std::tie(xadj, adjncy) = gccl::ReadGraph(input_graph);
        n_nodes = xadj.size() - 1;
        n_edges = xadj.back();
      }
    }
  }

  auto begin_part = GetTime();
  if(cached_graph_dir.size() > 0) {
    gccl::PartitionGraph(comm, cached_graph_dir.c_str(),&info, &my_n_nodes, &my_xadj, &my_adjncy);
  }
  else {
    gccl::PartitionGraph(comm, n_nodes, xadj.data(), adjncy.data(), &info,
                         &my_n_nodes, &my_xadj, &my_adjncy);
  }
  auto end_part = GetTime();
  LOG(INFO) << "Using time for partitioning graph "
            << TimeDiff(begin_part, end_part) << " ms";

  LOG(INFO) << "My n nodes " << my_n_nodes;
  int *input;
  std::vector<int> cpu_input(my_n_nodes * feat_size, 1);
  cudaMalloc((void **)&input, my_n_nodes * feat_size * sizeof(int));
  cudaMemcpy(input, cpu_input.data(), my_n_nodes * feat_size * sizeof(int),
             cudaMemcpyHostToDevice);

  comm->GetCoordinator()->Barrier();
  auto begin_ag = GetTime();
  int warm_up = 3;
  for (int i = 0; i < warm_up; ++i) {
    gccl::GraphAllgather(comm, info, input, gccl::gcclInt, feat_size, 0);
  }
  cudaStreamSynchronize(0);
  auto end_ag = GetTime();
  LOG(INFO) << "Using time for allgather in warm up "
            << TimeDiff(begin_ag, end_ag) / warm_up << " us";
  int iter = 10;
  begin_ag = GetTime();
  for (int i = 0; i < iter; ++i) {
    gccl::GraphAllgather(comm, info, input, gccl::gcclInt, feat_size, 0);
  }
  cudaStreamSynchronize(0);
  end_ag = GetTime();
  LOG(INFO) << "Using time for allgather in "
            << TimeDiff(begin_ag, end_ag) / iter << " us";
  for(int i = 0; i < warm_up; ++i) {
    gccl::GraphAllgatherBackward(comm, info, input, gccl::gcclFloat, feat_size, 0);
  }
  cudaStreamSynchronize(0);
  begin_ag = GetTime();
  for(int i = 0; i < iter; ++i) {
    gccl::GraphAllgatherBackward(comm, info, input, gccl::gcclFloat, feat_size, 0);
  }
  cudaStreamSynchronize(0);
  end_ag = GetTime();
  LOG(INFO) << "Using time for allgather backward in "
            << TimeDiff(begin_ag, end_ag) / iter << " us";
}

int main(int argc, char *argv[]) {
  FLAGS_logtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  MPI_Init(&argc, &argv);
  run();
  MPI_Finalize();
  return 0;
}
