#include "gccl.h"

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "mpi.h"

#include "base/bin_stream.h"
#include "communicator.h"
#include "test_mpi_utils.h"

namespace gccl {
namespace {

class TestMPIInit : public testing::Test {};

TEST_F(TestMPIInit, CommInitRank) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  gcclUniqueId id;
  if (rank == 0) {
    id = GetUniqueId();
  }
  LOG(INFO) << "My rank is " << rank << ", world size is " << world_size;
  MPI_Bcast((void *)&id, sizeof(gcclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  gcclComm_t comm;
  CommInitRank(&comm, world_size, id, rank);
  int comm_rank = comm->GetRank();
  int comm_n_ranks = comm->GetNRanks();
  EXPECT_EQ(comm_rank, rank);
  EXPECT_EQ(comm_n_ranks, world_size);

  auto *coor = comm->GetCoordinator();
  auto bs = std::make_shared<BinStream>();
  *bs << rank;
  coor->SendBinstreamTo(-1, bs);
  std::vector<bool> vis(world_size);
  if (rank == 0) {
    for (int i = 0; i < world_size; ++i) {
      auto bs = coor->RootRecvBinStream();
      int val;
      *bs >> val;
      vis[val] = true;
    }
    for (auto v : vis) {
      EXPECT_TRUE(v);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace
}  // namespace gccl