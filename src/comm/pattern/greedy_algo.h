#pragma once

#include <vector>

#include "comm/pattern/cost_model.h"

namespace gccl {

struct NodeScatterDecision {
  int n_stages;
  std::vector<int> offset;
  std::vector<std::pair<int, int>>
      send_pairs;  // (i, u, v) at stage i, send to device v from device u;
};

class GreedyAlgo {
 public:
  /**
   *
   *
   *
   */
  static NodeScatterDecision MakeDecision(CostModel* cost_model,
                                          const std::vector<int>& req_devs,
                                          int n_devs);
};
}  // namespace gccl
