#include "comm/pattern/cost_model.h"

#include <cmath>

#include "glog/logging.h"

namespace gccl {

Cost ComputeEdgeCost(double n, TransportLevel type, double cf) {
  // TODO
  double c = 0;
  switch(type) {
    case NV2:
      c = n;
      break;
     case NV1:
      c = n * 1.5;
      break;
     case PCIE:
      c = n * 8;
      break;
     case QPI:
      c = n * 8;
      break;
     case IB_NET:
      c = n * 8;
      break;
     default:
      CHECK(false);
  };
  c *= cf;
  return c;
}

double sigmoid(double x) {
  if (x > 0) return x * 10;
  double ret = 1 / (1 + std::exp(-x));
  return ret;
}

bool CostModel::CanConnect(const Node& u, const Node& v) const {
  if (dev_graph_->GetTransportLevel(u, v) <= tr_lv_) {
    return true;
  }
  return false;
}

Cost CostModel::GetEdgeCost(int stage, const Node& u, const Node& v) const {
  DCHECK_GE(stage, 0);
  DCHECK_LT(stage, n_stages_);
  double max_inc = -100000000;
  Path path = dev_graph_->GetPath(u, v);
  CHECK(path.src != -1);
  for (const Edge& e : path.edges) {
    int id = e.id;
    int n = 0;
    if (edge_cost_[stage].count(id) > 0) {
      n = edge_cost_[stage].at(id);
    }
    double prev_cost = curr_cost_[stage];
    double inc = ComputeEdgeCost(n + 1, e.type, e.cf) - prev_cost;
    inc = sigmoid(inc);
    max_inc = std::max(max_inc, inc);
  }
  return max_inc;
}

void CostModel::IncEdgeCost(int stage, const Node& u, const Node& v) {
  DCHECK_GE(stage, 0);
  DCHECK_LT(stage, n_stages_);
  Path path = dev_graph_->GetPath(u, v);
  CHECK(path.src != -1);
  for (const Edge& e : path.edges) {
    int id = e.id;
    edge_cost_[stage][id]++;
    Cost c = ComputeEdgeCost(edge_cost_[stage][id], e.type, e.cf);
    if (c > curr_cost_[stage]) {
      curr_cost_[stage] = c;
    }
  }
}

}  // namespace gccl
