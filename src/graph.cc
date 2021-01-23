#include "graph.h"

#include <algorithm>
#include <fstream>
#include <random>

#include "glog/logging.h"

#include "gccl.h"
#include "utils.h"

namespace gccl {

Graph::Graph(int n, const std::vector<std::pair<int, int>> &edges) {
  n_nodes = n;
  n_edges = edges.size();
  xadj.push_back(0);
  auto sorted_edges = edges;
  std::sort(sorted_edges.begin(), sorted_edges.end(),
            [](const auto &x, const auto &y) {
              if (x.first != y.first) return x.first < y.first;
              return x.second < y.second;
            });
  for (int i = 0, j = 0; i < n; ++i) {
    while (j < sorted_edges.size() && sorted_edges[j].first == i) {
      adjncy.push_back(sorted_edges[j].second);
      ++j;
    }
    xadj.push_back(adjncy.size());
  }
}

int FindMaxIdx(const std::vector<std::pair<int, int>> &edges) {
  int ret = 0;
  for (const auto &p : edges) {
    ret = std::max(ret, p.first);
    ret = std::max(ret, p.second);
  }
  return ret;
}

Graph::Graph(const std::vector<std::pair<int, int>> &edges)
    : Graph(FindMaxIdx(edges) + 1, edges) {}

std::vector<std::pair<int, int>> BuildEdges(int n, int *xadj, int *adjncy) {
  std::vector<std::pair<int, int>> edges;
  for (int i = 0; i < n; ++i) {
    for (int j = xadj[i]; j < xadj[i + 1]; ++j) {
      edges.push_back({i, adjncy[j]});
    }
  }
  return edges;
}

Graph::Graph(int n, int *xadj, int *adjncy)
    : Graph(n, BuildEdges(n, xadj, adjncy)) {}

Graph::Graph(const std::string &file) {
  auto csr_vec = ReadGraph(file);
  int n = csr_vec.first.size() - 1;
  *this = Graph(n, csr_vec.first.data(), csr_vec.second.data());
}

void Graph::WriteToFile(const std::string &file) const {
  std::ofstream out(file);
  CHECK(out.is_open()) << "Cannot open graph file " << file;
  out << n_nodes << ' ' << n_edges << '\n';
  auto edge_func = [&out](int u, int v) { out << u << ' ' << v << '\n'; };
  ApplyEdge(edge_func);
}

BinStream &Graph::serialize(BinStream &bs) const {
  bs << n_nodes << n_edges << xadj << adjncy;
  return bs;
}
BinStream &Graph::deserialize(BinStream &bs) {
  bs >> n_nodes >> n_edges >> xadj >> adjncy;
  return bs;
}

void BuildRawGraph(const Graph &g, int *n, int **xadj, int **adjncy) {
  *n = g.n_nodes;
  CopyVectorToRawPtr(xadj, g.xadj);
  CopyVectorToRawPtr(adjncy, g.adjncy);
}

std::pair<std::vector<int>, std::vector<int>> BuildGraphFromEdges(
    int n, std::vector<std::pair<int, int>> &edges) {
  std::sort(edges.begin(), edges.end(),
            [](const std::pair<int, int> &lhs, const std::pair<int, int> &rhs) {
              if (lhs.first != rhs.first) return lhs.first < rhs.first;
              return lhs.second < rhs.second;
            });

  std::vector<int> xadj, adjncy;
  xadj.push_back(0);
  int j = 0;
  for (int i = 0; i < n; ++i) {
    while (j < edges.size() && edges[j].first == i) {
      adjncy.push_back(edges[j].second);
      ++j;
    }
    xadj.push_back(j);
  }
  CHECK(j == edges.size());
  return {xadj, adjncy};
}

std::pair<std::vector<int>, std::vector<int>> RandGraph(int n, int m) {
  // RandomGenerator rg(0);
  std::mt19937 gen(0);
  std::uniform_int_distribution<> dist(0, n - 1);
  std::vector<std::pair<int, int>> edges;
  for (int i = 0; i < m; ++i) {
    int u = dist(gen);
    int v = dist(gen);
    edges.push_back({u, v});
  }
  return BuildGraphFromEdges(n, edges);
}

std::pair<std::vector<int>, std::vector<int>> ReadGraph(
    const std::string &file) {
  LOG(INFO) << "Trying to load graph on " << file;
  std::ifstream stream;
  stream.open(file);
  std::vector<std::pair<int, int>> edges;
  int n, m;
  stream >> n >> m;
  std::vector<int> san;
  LOG(INFO) << "N nodes is " << n << " n edges is " << m;
  for (int i = 0; i < m; ++i) {
    int u, v;
    stream >> u >> v;
    edges.push_back({u, v});
    san.push_back(u);
    san.push_back(v);
  }
  std::sort(san.begin(), san.end());
  san.erase(std::unique(san.begin(), san.end()), san.end());
  CHECK_LE(san.size(), n);
  for (auto &e : edges) {
    e.first = std::lower_bound(san.begin(), san.end(), e.first) - san.begin();
    e.second = std::lower_bound(san.begin(), san.end(), e.second) - san.begin();
  }
  LOG(INFO) << "Finished loading";
  return BuildGraphFromEdges(n, edges);
}

}  // namespace gccl
