#pragma once

#include <vector>

#include "base/bin_stream.h"

namespace gccl {

struct Graph {
  template <typename EdgeFunc>
  void ApplyEdge(EdgeFunc func) const {
    for (int i = 0; i < n_nodes; ++i) {
      for (int j = xadj[i]; j < xadj[i + 1]; ++j) {
        func(i, adjncy[j]);
      }
    }
  }
  // should be deprecated
  Graph() {}
  Graph(int n, const std::vector<std::pair<int, int>> &edges);
  Graph(const std::vector<std::pair<int, int>> &edges);
  Graph(int n, int *xadj, int *adjncy);
  Graph(const std::string &file);

  void WriteToFile(const std::string &file) const;

  BinStream &serialize(BinStream &bs) const;
  BinStream &deserialize(BinStream &bs);

  std::vector<int> xadj;    // offset
  std::vector<int> adjncy;  // neighbours
  int n_nodes;              // n_nodes_ + 1 == xadj_.size()
  int n_edges;              // n_edges_ == adjncy_.size()
};

struct LocalGraphInfo {
  int n_nodes;
  int n_local_nodes;
};

void BuildRawGraph(const Graph &g, int *n, int **xadj, int **adjncy);

}  // namespace gccl
