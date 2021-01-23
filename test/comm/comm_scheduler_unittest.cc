#include "comm/comm_scheduler.h"

#include <memory>

#include "gtest/gtest.h"

#include "test_utils.h"

namespace gccl {
namespace {

class TestCommScheduler : public testing::Test {
 public:
  TestCommScheduler() {
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

TEST_F(TestCommScheduler, BuildLocalMappings) {
  CommScheduler sch;
  sch.BuildLocalMappings(*g1, n_parts1, parts1);
  auto local_mappings = sch.GetLocalMappings();
  EXPECT_MAP_EQ(local_mappings[0], {{0, 0}, {1, 1}, {3, 2}});
  EXPECT_MAP_EQ(local_mappings[1], {{2, 0}, {3, 1}, {1, 2}, {4, 3}, {5, 4}});
  EXPECT_MAP_EQ(local_mappings[2], {{4, 0}, {5, 1}, {0, 2}, {2, 3}});
}

TEST_F(TestCommScheduler, BuildTransferRequest) {
  CommScheduler sch;
  auto req = sch.BuildTransferRequest(*g1, n_parts1, parts1);
  EXPECT_VEC_EQ(req.req_ids[0][1], {1});
  EXPECT_VEC_EQ(req.req_ids[0][2], {0});
  EXPECT_VEC_EQ(req.req_ids[1][0], {3});
  EXPECT_VEC_EQ(req.req_ids[1][2], {2});
  EXPECT_VEC_EQ(req.req_ids[2][0], {});
  EXPECT_VEC_EQ(req.req_ids[2][1], {4, 5});
}

void EXPECT_GRAPH_EQ(const Graph& lhs, const Graph& rhs) {
  EXPECT_EQ(lhs.n_nodes, rhs.n_nodes);
  EXPECT_EQ(lhs.n_edges, rhs.n_edges);
  EXPECT_VEC_EQ(lhs.xadj, rhs.xadj);
  EXPECT_VEC_EQ(lhs.adjncy, rhs.adjncy);
}

TEST_F(TestCommScheduler, BuildSubgraphs) {
  CommScheduler sch;
  sch.BuildLocalMappings(*g1, n_parts1, parts1);
  std::vector<Graph> subgraphs =
      sch.BuildSubgraphs(*g1, sch.GetLocalMappings(), parts1, 3);
  EXPECT_GRAPH_EQ(subgraphs[0], Graph({{0, 1}, {2, 0}}));
  EXPECT_GRAPH_EQ(subgraphs[1], Graph({{2, 0}, {4, 0}, {3, 1}}));
  EXPECT_GRAPH_EQ(subgraphs[2], Graph({{2, 1}, {3, 0}}));
}

/*
TEST_F(TestCommScheduler, AllocateRequestToBlock) {
  CommScheduler sch;
  auto req = sch.BuildTransferRequest(*g1, n_parts1, parts1);
  auto per_block_reqs = sch.AllocateRequestToBlock(req, n_parts1, 2);
  EXPECT_VEC_EQ(per_block_reqs[0].req_ids[0][2], {0});
  EXPECT_VEC_EQ(per_block_reqs[0].req_ids[1][2], {2});
  EXPECT_VEC_EQ(per_block_reqs[0].req_ids[2][1], {4});

  EXPECT_VEC_EQ(per_block_reqs[1].req_ids[0][1], {1});
  EXPECT_VEC_EQ(per_block_reqs[1].req_ids[1][0], {3});
  EXPECT_VEC_EQ(per_block_reqs[1].req_ids[2][1], {5});
  int cnt1 = 0, cnt2 = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (per_block_reqs[0].req_ids[i][j].size() > 0) cnt1++;
      if (per_block_reqs[1].req_ids[i][j].size() > 0) cnt2++;
    }
  }
  EXPECT_EQ(cnt1, 3);
  EXPECT_EQ(cnt2, 3);
}
*/

}  // namespace
}  // namespace gccl
