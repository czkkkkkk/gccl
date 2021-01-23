#include "comm/pattern/comm_pattern.h"

#include <memory>

#include "gtest/gtest.h"

#include "comm/comm_scheduler.h"
#include "comm/pattern/ring_comm_pattern.h"
#include "config.h"
#include "graph.h"
#include "test_utils.h"
#include "utils.h"

namespace gccl {
namespace {

class TestRingCommPattern : public testing::Test {
 public:
  TestRingCommPattern() {
    g1 = std::make_shared<Graph>(
        6, std::vector<std::pair<int, int>>(
               {{0, 1}, {0, 5}, {1, 2}, {2, 4}, {3, 0}, {4, 3}, {5, 2}}));
    parts1 = {0, 0, 1, 1, 2, 2};
    n_parts1 = 3;
  }

 protected:
  std::shared_ptr<Graph> g1;
  int n_parts1;
  std::vector<int> parts1;
};

TEST_F(TestRingCommPattern, BuildCommPatternInfos) {
  std::vector<ConnType> conn_type(3, ConnType::P2P);
  RingCommPattern pattern({0, 1, 2}, conn_type);
  Config config = DefaultConfig(3);
  CommScheduler sch;
  sch.BuildLocalMappings(*g1, n_parts1, parts1);
  auto req = sch.BuildTransferRequest(*g1, n_parts1, parts1);
  auto comm_infos = pattern.BuildCommPatternInfos(
      &config, sch.GetLocalMappings(), req, n_parts1);

  std::vector<RingCommPatternInfo*> ring_infos;
  for (auto& info : comm_infos) {
    ring_infos.push_back(info.GetRingCommPatternInfo());
  }

  EXPECT_EQ(ring_infos[0]->n_stages, 2);
  EXPECT_VEC_EQ(ring_infos[0]->send_ids, ring_infos[0]->send_ids + 4,
                {1, 0, -1, -2});
  EXPECT_VEC_EQ(ring_infos[0]->send_off, ring_infos[0]->send_off + 3,
                {0, 2, 4});
  EXPECT_VEC_EQ(ring_infos[0]->recv_ids, ring_infos[0]->recv_ids + 3,
                {-1, -2, 2});
  EXPECT_VEC_EQ(ring_infos[0]->recv_off, ring_infos[0]->recv_off + 3,
                {0, 2, 3});

  EXPECT_EQ(ring_infos[1]->n_stages, 2);
  EXPECT_VEC_EQ(ring_infos[1]->send_ids, ring_infos[1]->send_ids + 3,
                {0, 1, -1});
  EXPECT_VEC_EQ(ring_infos[1]->send_off, ring_infos[1]->send_off + 3,
                {0, 2, 3});
  EXPECT_VEC_EQ(ring_infos[1]->recv_ids, ring_infos[1]->recv_ids + 4,
                {2, -1, 3, 4});
  EXPECT_VEC_EQ(ring_infos[1]->recv_off, ring_infos[1]->recv_off + 3,
                {0, 2, 4});

  EXPECT_EQ(ring_infos[2]->n_stages, 2);
  EXPECT_VEC_EQ(ring_infos[2]->send_ids, ring_infos[2]->send_ids + 3,
                {0, 1, -1});
  EXPECT_VEC_EQ(ring_infos[2]->send_off, ring_infos[2]->send_off + 3,
                {0, 2, 3});
  EXPECT_VEC_EQ(ring_infos[2]->recv_ids, ring_infos[2]->recv_ids + 3,
                {3, -1, 2});
  EXPECT_VEC_EQ(ring_infos[2]->recv_off, ring_infos[2]->recv_off + 3,
                {0, 2, 3});
}

}  // namespace
}  // namespace gccl