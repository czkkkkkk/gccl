#include "graph.h"

#include <memory>

#include "gtest/gtest.h"

#include "test_utils.h"

namespace gccl {
namespace {

class TestGraph : public testing::Test {
 public:
  TestGraph() {
    edges = {{0, 1}, {1, 2}, {2, 0}};
    g = std::make_shared<Graph>(3, edges);
  }

 protected:
  std::shared_ptr<Graph> g;
  std::vector<std::pair<int, int>> edges;
};

TEST_F(TestGraph, Ctr) {
  Graph *g = new Graph(3, {{0, 1}, {2, 0}, {1, 2}, {0, 2}, {0, 1}});
  EXPECT_NE(g, nullptr);
  EXPECT_EQ(g->n_nodes, 3);
  EXPECT_EQ(g->n_edges, 5);
  EXPECT_VEC_EQ(g->xadj, {0, 3, 4, 5});
  EXPECT_VEC_EQ(g->adjncy, {1, 1, 2, 2, 0});
}

TEST_F(TestGraph, ApplyEdge) {
  std::vector<std::pair<int, int>> edges_in_graph;
  g->ApplyEdge([&edges_in_graph](int u, int v) {
    edges_in_graph.push_back({u, v});
  });
  EXPECT_VEC_EQ(edges_in_graph, edges);
}

TEST_F(TestGraph, Serialization) {
  Graph g({{0, 1}, {2, 0}, {1, 2}, {0, 2}, {0, 1}});
  BinStream bs;
  bs << g;
  Graph ng;
  bs >> ng;
  EXPECT_EQ(g.n_nodes, ng.n_nodes);
  EXPECT_EQ(g.n_edges, ng.n_edges);
  EXPECT_VEC_EQ(g.xadj, ng.xadj);
  EXPECT_VEC_EQ(g.adjncy, ng.adjncy);
}

TEST_F(TestGraph, BuildRawGraph) {
  Graph g({{0, 1}, {2, 0}, {1, 2}, {0, 2}, {0, 1}});
  int n, *xadj, *adjncy;
  BuildRawGraph(g, &n, &xadj, &adjncy);
  EXPECT_EQ(n, g.n_nodes);
  EXPECT_VEC_EQ(xadj, g.xadj);
  EXPECT_VEC_EQ(adjncy, g.adjncy);
  delete xadj, adjncy;
}

}  // namespace
}  // namespace gccl