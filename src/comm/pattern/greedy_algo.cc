#include "comm/pattern/greedy_algo.h"

#include <algorithm>
#include <limits>
#include <set>
#include <queue>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "utils.h"
#include "param.h"

namespace gccl {

void CheckValid(
  const std::vector<std::tuple<int, int, int>>& trs, const std::vector<int>& req_devs, int n_devs) {
  //LOG(INFO) << "-------------";
  //LOG(INFO) << "Req devs are " << VecToString(req_devs);
  //for (auto p : trs) {
  //  int s = std::get<0>(p);
  //  int u = std::get<1>(p);
  //  int v = std::get<2>(p);
  //  LOG(INFO) << "Stage " << s << " from " << u << " to " << v;
  //}
  std::vector<bool> vis(n_devs, false);
  vis[req_devs[0]] = true;
  std::set<int> req_devs_set(req_devs.begin(), req_devs.end());
  int cnt = 1;
  int stage = 0;
  for(int i = 0, j = 0; i < trs.size(); i = j) {
    CHECK_EQ(stage, std::get<0>(trs[i]));
    while(j < trs.size() && std::get<0>(trs[j]) == stage) {
      int u = std::get<1>(trs[j]);
      int v = std::get<2>(trs[j]);
      CHECK(vis[u]);
      CHECK(!vis[v]);
      ++j;
    }
    for(int k = i; k < j; ++k) {
      int u = std::get<1>(trs[k]);
      int v = std::get<2>(trs[k]);
      CHECK(!vis[v]);
      if(req_devs_set.count(v) > 0) {
        cnt++;
      }
      vis[v] = 1;
    }
    stage ++;
  }
  CHECK(cnt == req_devs.size());
}

int dcmp(double x) { 
  if(fabs(x) < 1e-4) return 0;
  return x < 0? -1: 1;
}

NodeScatterDecision GreedyAlgo::MakeDecision(CostModel* cost_model,
                                             const std::vector<int>& req_devs,
                                             int n_devs) {
  std::vector<std::tuple<int, int, int>> trs;

  std::vector<bool> reach(n_devs, false);
  std::vector<int> tree_lv(n_devs, -1);
  std::set<int> req_devs_set(req_devs.begin(), req_devs.end());
  int src = req_devs.front();
  reach[src] = true;
  tree_lv[src] = 0;
  for (int i = 0; i < req_devs.size() - 1; ++i) {
    std::vector<double> dist(n_devs, std::numeric_limits<double>::max());
    std::vector<int> prev(n_devs, -1);
    std::vector<bool> vis(n_devs, false);
    auto pre_reach = reach;
    std::priority_queue<std::pair<double, int>> q;
    for(int u = 0; u < n_devs; ++u) {
      if (reach[u]) {
        dist[u] = 0;
        vis[u] = true;
        q.push({0, u});
      } else {
        tree_lv[u] = -1;
      }
    }
    // dijkstra 
    while (!q.empty()) {
      double d;
      int u;
      std::tie(d, u) = q.top();
      q.pop();
      if(dcmp(-d - dist[u]) > 0) continue;
      vis[u] = true;
      if (!reach[u] && req_devs_set.count(u) > 0) {
        continue;
      }

      int lv = tree_lv[u];
      for (int v = 0; v < n_devs; ++v) {
        if(!cost_model->CanConnect(u, v) || reach[v] || vis[v]) continue;
        // if(!cost_model->CanConnect(u, v) || reach[v] || vis[v] || req_devs_set.count(v) == 0) continue;
        int max_depth = GetEnvParam<int>("MAX_DEPTH", n_devs);
        if(lv == n_devs || lv >= max_depth) {
          continue;
        }
        double c = cost_model->GetEdgeCost(lv, u, v) + 1e-4;
        if (dist[v] > dist[u] + c) {
          prev[v] = u;
          dist[v] = dist[u] + c;
          tree_lv[v] = tree_lv[u] + 1;
          q.push({-dist[v], v});
        }
      }
    }  // dijkstra 
    int nearest_dev = -1;
    double min_dist = std::numeric_limits<double>::max();
    for (auto u : req_devs) {
      if (!reach[u] && dist[u] < min_dist) {
        nearest_dev = u;
        min_dist = dist[u];
      }
    }
    if(nearest_dev == -1) {
      LOG(INFO) << "# req devs are " << VecToString(req_devs);
      LOG(INFO) << "# dist are " << VecToString(dist);
      LOG(INFO) << "# reach are " << VecToString(reach);
      LOG(INFO) << "# pre reach are " << VecToString(pre_reach);
      LOG(INFO) << "# prevs are " << VecToString(prev);
    }
    CHECK(nearest_dev != -1) << "Some request device is not reachable";
    int t = nearest_dev;
    while (prev[t] != -1) {
      int stage = tree_lv[prev[t]];
      trs.push_back(std::tuple<int, int, int>(stage, prev[t], t));
      cost_model->IncEdgeCost(stage, prev[t], t);
      if(t != nearest_dev && req_devs_set.count(t) > 0) {
        CHECK(false);
      }
      reach[t] = true;
      t = prev[t];
    }
  }
  // Note: comparison of tuple is removed in c++20
  std::sort(trs.begin(), trs.end());
  // CheckValid(trs, req_devs, n_devs);

  NodeScatterDecision ret;
  CHECK_LT(trs.size(), n_devs);
  ret.n_stages = trs.size() == 0? 0: (std::get<0>(trs.back()) + 1);
  ret.offset.push_back(0);
  for (int stage = 0, j = 0; stage < ret.n_stages; ++stage) {
    while (j < trs.size() && std::get<0>(trs[j]) == stage) {
      int u = std::get<1>(trs[j]);
      int v = std::get<2>(trs[j]);
      ret.send_pairs.push_back({u, v});
      ++j;
    }
    ret.offset.push_back(ret.send_pairs.size());
  }
  return ret;
}

}  // namespace gccl
