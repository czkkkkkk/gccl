#pragma once

#include <map>

#include "topo/dev_graph.h"

namespace gccl {

typedef double Cost;

class CostModel {
 public:
  CostModel(DevGraph* dev_graph, TransportLevel lv, int n_stages)
      : dev_graph_(dev_graph), tr_lv_(lv) {
    curr_cost_.resize(n_stages, 0);
    n_stages_ = n_stages;
    edge_cost_.resize(n_stages);
  }

  bool CanConnect(const Node& u, const Node& v) const;
  Cost GetEdgeCost(int stage, const Node& u, const Node& v) const;
  void IncEdgeCost(int stage, const Node& u, const Node& v);
  TransportLevel GetTransportLevel() const { return tr_lv_; }

 private:
  int n_stages_;
  TransportLevel tr_lv_;
  DevGraph* dev_graph_;
  std::vector<Cost> curr_cost_;
  std::vector<std::map<int, Cost>> edge_cost_;
  std::vector<std::map<int, Cost>> node_cost_;
};

}  // namespace gccl
