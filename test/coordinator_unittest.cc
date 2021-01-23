#include "coordinator.h"

#include <thread>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "zmq.hpp"

#include "test_utils.h"

namespace gccl {
namespace {

class TestCoordinator : public testing::Test {
 public:
  TestCoordinator() {
    world_size_ = 3;
    ctx_.resize(world_size_);
    for (int i = 0; i < world_size_; ++i) {
      coor_.emplace_back(&ctx_[i]);
    }
    for (int i = 0; i < world_size_; ++i) {
      coor_[i].SetRankAndNPeers(i, world_size_);
    }
    gcclUniqueId id = StringToUniqueId("tcp://127.0.0.1:12322");
    coor_[0].RootInit(id);
    MultiThreading(world_size_,
                   [this, &id](int rank) { coor_[rank].BuildPeerInfo(id); });
  }

 protected:
  int world_size_;
  std::vector<zmq::context_t> ctx_;
  std::vector<Coordinator> coor_;
};

TEST_F(TestCoordinator, CtrAndDctr) {
  zmq::context_t ctx;
  Coordinator *coor = new Coordinator(&ctx);
  EXPECT_TRUE(coor != nullptr);
  delete coor;
}

TEST_F(TestCoordinator, Setup) {
  for (int i = 0; i < world_size_; ++i) {
    auto bs = std::make_shared<BinStream>();
    *bs << i;
    coor_[i].SendBinstreamTo(-1, bs);
  }
  std::vector<bool> vis(world_size_, false);
  for (int i = 0; i < world_size_; ++i) {
    auto bs = coor_[0].RootRecvBinStream();
    int val;
    *bs >> val;
    vis[val] = true;
  }
  for (auto v : vis) {
    EXPECT_TRUE(v);
  }
}

struct TestStructure {
  int val;
};

TEST_F(TestCoordinator, Allgather) {
  std::vector<std::thread> ths;
  for (int i = 0; i < world_size_; ++i) {
    ths.emplace_back([this, rank = i]() {
      std::vector<TestStructure> vecs(world_size_);
      vecs[rank].val = rank;
      coor_[rank].Allgather(vecs);
      for (int i = 0; i < world_size_; ++i) {
        EXPECT_EQ(i, vecs[i].val);
      }
    });
  }
  for (auto &t : ths) t.join();
}

TEST_F(TestCoordinator, Scatter) {
  std::vector<std::thread> ths;
  for (int i = 0; i < world_size_; ++i) {
    ths.emplace_back([this, rank = i]() {
      std::vector<TestStructure> vec(world_size_);
      if (rank == 0) {
        for (int j = 0; j < world_size_; ++j) {
          vec[j].val = j;
        }
      }
      auto my_info = coor_[rank].Scatter(vec);
      EXPECT_EQ(my_info.val, rank);
    });
  }
  for (auto &t : ths) t.join();
}

TEST_F(TestCoordinator, Broadcast) {
  MultiThreading(world_size_, [this](int rank) {
    std::vector<int> vec;
    if (rank == 0) {
      vec = std::vector<int>(3, 1);
    }
    coor_[rank].Broadcast(vec);
    EXPECT_VEC_EQ(vec, std::vector<int>(3, 1));
  });
}

TEST_F(TestCoordinator, RingExchange) {
  MultiThreading(world_size_, [this](int rank) {
    int next = (rank + 2) % world_size_;
    int recv_val = coor_[rank].RingExchange(next, rank);
    EXPECT_EQ(recv_val, next);
  });
}

}  // namespace
}  // namespace gccl
