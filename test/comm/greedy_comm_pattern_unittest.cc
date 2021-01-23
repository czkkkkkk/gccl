#include "comm/pattern/greedy_comm_pattern.h"

#include <memory>

#include "gtest/gtest.h"

#include "comm/comm_scheduler.h"
#include "config.h"
#include "graph.h"
#include "test_utils.h"
#include "utils.h"

namespace gccl {
namespace {

class TestGreedyCommPattern : public testing::Test {
 public:
  TestGreedyCommPattern() {
    g1 = std::make_shared<Graph>(
        7, std::vector<std::pair<int, int>>(
               {{0, 1}, {0, 2}, {1, 3}, {1, 4}, {2, 5}, {2, 6}}));
    parts1 = {0, 1, 2, 0, 1, 2, 0};
    n_parts1 = 3;
  }

 protected:
  std::shared_ptr<Graph> g1;
  int n_parts1;
  std::vector<int> parts1;
};

TEST_F(TestGreedyCommPattern, BuildCommPatternInfos) {
  std::vector<ConnType> conn_type(3, ConnType::P2P);
  GreedyCommPattern pattern({0, 1, 2}, conn_type);
  Config config = DefaultConfig(3);
  CommScheduler sch;
  sch.BuildLocalMappings(*g1, n_parts1, parts1);
  auto req = sch.BuildTransferRequest(*g1, n_parts1, parts1);
  auto comm_infos = pattern.BuildCommPatternInfos(
      &config, sch.GetLocalMappings(), req, n_parts1);

  std::vector<GreedyCommPatternInfo*> aa_infos;
  for (auto& info : comm_infos) {
    aa_infos.push_back(info.GetGreedyCommPatternInfo());
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(aa_infos[i]->n_peers, 3);
    EXPECT_EQ(aa_infos[i]->rank, i);
    EXPECT_EQ(aa_infos[i]->n_stages, 1);
    EXPECT_EQ(aa_infos[i]->extra_buffer_size, 1);
    EXPECT_VEC_EQ(aa_infos[i]->max_comm_size, {1});
    if (i == 0) {
      EXPECT_VEC_EQ(aa_infos[i]->send_ids, {0, 0});
      EXPECT_VEC_EQ(aa_infos[i]->send_off, {0, 0, 1, 2});
      EXPECT_VEC_EQ(aa_infos[i]->recv_ids, {3, 4});
      EXPECT_VEC_EQ(aa_infos[i]->recv_off, {0, 0, 1, 2});
    } else {
      EXPECT_VEC_EQ(aa_infos[i]->send_ids, {0});
      EXPECT_VEC_EQ(aa_infos[i]->send_off, {0, 1, 1, 1});
      EXPECT_VEC_EQ(aa_infos[i]->recv_ids, {2});
      EXPECT_VEC_EQ(aa_infos[i]->recv_off, {0, 1, 1, 1});
    }
  }
}

}  // namespace
}  // namespace gccl
