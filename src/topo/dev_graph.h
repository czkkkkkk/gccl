#pragma once

#include <string>
#include <vector>

namespace gccl {

enum TransportLevel { NV2, NV1, PCIE, QPI, IB_NET, UNCONNECT };

typedef int Node;

TransportLevel StringToTransportLevel(const std::string& type);

struct Edge {
  int id;
  double cf;
  TransportLevel type;
  Node u, v;
};

struct Path {
  TransportLevel max_tr_level;
  Node src, dst;
  std::vector<Edge> edges;
};

class DevGraph {
 public:
  DevGraph(const std::string& json_file);
  DevGraph(int n_devs, int n_virtual_devs, const std::vector<Edge>& edges);

  TransportLevel GetTransportLevel(Node u, Node v) const;
  Path GetPath(Node u, Node v) const;

 private:
  void InitializeEdges(std::vector<Edge>& edges, const std::vector<std::pair<int, int>>& unconn_edges);

  int n_devs_, n_virtual_devs_;
  std::vector<std::vector<Edge>> edges_;  // Peer to peer edges
  std::vector<std::vector<Path>> paths_;  // Peer to peer path
};

DevGraph DefaultDevGraph(int n_devs);

}  // namespace gccl
