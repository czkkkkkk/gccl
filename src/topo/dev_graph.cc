#include "topo/dev_graph.h"

#include <fstream>
#include <queue>
#include <algorithm>

#include "nlohmann/json.hpp"
#include "glog/logging.h"

#include "utils.h"

namespace gccl {

using json = nlohmann::json;

TransportLevel StringToTransportLevel(const std::string& type) {
  TransportLevel ret;
  if (type == "NV2") {
    ret = NV2;
  } else if (type == "NV1") {
    ret = NV1;
  } else if (type == "PCIE") {
    ret = PCIE;
  } else if (type == "QPI") {
    ret = QPI;
  } else if (type == "IB_NET") {
    ret = IB_NET;
  } else {
    // TODO
    CHECK(false) << "Unsupported connection level " << type;
  }
  return ret;
}


void DevGraph::InitializeEdges(std::vector<Edge>& edges, const std::vector<std::pair<int, int>>& unconn_edges) {
  int org_edges_size = edges.size();
  edges.resize(org_edges_size * 2);
  for (int i = 0; i < org_edges_size; ++i) {
    edges[i].id = i;
    edges[i + org_edges_size] = edges[i];
    std::swap(edges[i + org_edges_size].u, edges[i + org_edges_size].v);
    edges[i + org_edges_size].id = i + org_edges_size;
  }

  int total_devices = n_devs_ + n_virtual_devs_;
  edges_.resize(total_devices);
  paths_.resize(total_devices);
  for (auto& adj : edges_) {
    adj.resize(total_devices);
    for (auto& e : adj) {
      e.id = -1;
      e.type = UNCONNECT;
    }
  }
  for (auto& peer_paths : paths_) {
    peer_paths.resize(total_devices);
    for (auto& p : peer_paths) {
      p.max_tr_level = UNCONNECT;
    }
  }
  std::vector<std::vector<bool>> unconn_map(n_devs_, std::vector<bool>(n_devs_, false));
  for(const auto& p: unconn_edges) {
    int u = p.first, v = p.second;
    unconn_map[u][v] = unconn_map[v][u] = true;
  }
  for (const auto& e : edges) {
    // TODO
    int u = e.u, v = e.v;
    edges_[u][v] = e;
  }
  auto bfs_func = [this, total_devices](int s, std::vector<Edge>& prev) {
    std::queue<int> q;
    q.push(s);
    std::vector<bool> vis(total_devices, false);
    vis[s] = true;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      for (int v = 0; v < total_devices; ++v) {
        auto& e = edges_[u][v];
        if (e.type == UNCONNECT || e.type == NV2 || e.type == NV1 || vis[v]) {
          continue;
        }
        vis[v] = true;
        prev[v] = e;
        q.push(v);
      }
    }
  };
  auto get_path = [](int u, int v, const std::vector<Edge>& prev) {
    if(u == v) return std::vector<Edge>();
    CHECK(prev[v].id != -1);
    std::vector<Edge> ret;
    while (prev[v].u != -1) {
      ret.push_back(prev[v]);
      v = prev[v].u;
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
  };
  auto get_tr_level = [](const std::vector<Edge>& path) {
    TransportLevel ret = NV2;
    for (const auto& e : path) {
      if (e.type > ret) {
        ret = e.type;
      }
    }
    return ret;
  };

  for (int u = 0; u < n_devs_; ++u) {
    std::vector<Edge> prev(n_devs_ + n_virtual_devs_, Edge{-1, 1, UNCONNECT, -1, -1});
    bfs_func(u, prev);
    for (int v = 0; v < n_devs_; ++v) {
      auto& p = paths_[u][v];
      auto& e = edges_[u][v];
      if (e.type == NV2 || e.type == NV1) {
        p.max_tr_level = e.type;
        p.src = u;
        p.dst = v;
        p.edges = {e};
      }
      else if(unconn_map[u][v] || prev[v].id == -1) {
        p.max_tr_level = UNCONNECT;
        p.src = p.dst = -1;
        p.edges = {};
      } else  {
        // p2p can only be connected by nvlink
        CHECK(e.type == UNCONNECT);
        std::vector<Edge> path = get_path(u, v, prev);
        p.max_tr_level = get_tr_level(path);
        p.src = u;
        p.dst = v;
        p.edges = path;
      }
    }
  }
}

DevGraph::DevGraph(const std::string& json_file) {
  std::ifstream input(json_file);
  json json_graph;
  input >> json_graph;
  json_graph.at("n_devs").get_to(n_devs_);
  json_graph.at("n_virtual_devs").get_to(n_virtual_devs_);
  std::vector<int> src;
  std::vector<int> dst;
  std::vector<std::string> types;
  std::vector<double> contention_factor;
  json_graph.at("edges").at("src").get_to(src);
  json_graph.at("edges").at("dst").get_to(dst);
  json_graph.at("edges").at("type").get_to(types);
  std::vector<std::pair<int, int>> unconn_edges;
  if(json_graph.at("edges").contains("contention_factor")) {
    json_graph.at("edges").at("contention_factor").get_to(contention_factor);
  } else {
    contention_factor.resize(src.size(), 1);
  }
  if(json_graph.contains("unconn_edges")) {
    json_graph.at("unconn_edges").get_to(unconn_edges);
  }
  std::vector<Edge> edges;
  CHECK(src.size() == dst.size() && src.size() == types.size());
  for (int i = 0; i < src.size(); ++i) {
    Edge e;
    e.u = src[i];
    e.v = dst[i];
    e.type = StringToTransportLevel(types[i]);
    e.cf = contention_factor[i];
    edges.push_back(e);
  }

  InitializeEdges(edges, unconn_edges);
}

DevGraph::DevGraph(int n_devs, int n_virtual_devs,
                   const std::vector<Edge>& edges) {
  n_devs_ = n_devs;
  n_virtual_devs_ = n_virtual_devs;
  auto mutable_edges = edges;
  InitializeEdges(mutable_edges, {});
}

TransportLevel DevGraph::GetTransportLevel(Node u, Node v) const {
  if(edges_[u][v].type == NV2 || edges_[u][v].type == NV1) {
    return edges_[u][v].type;
  }
  return paths_[u][v].max_tr_level;
}

Path DevGraph::GetPath(Node u, Node v) const { return paths_[u][v]; }

DevGraph DefaultDevGraph(int n_devs) {
  // Ring connection
  std::vector<Edge> edges(n_devs);
  for (int i = 0; i < n_devs; ++i) {
    edges[i].u = i;
    edges[i].v = (i + 1) % n_devs;
    edges[i].id = i;
    edges[i].cf = 1;
    edges[i].type = NV2;
  }
  return DevGraph(n_devs, 0, edges);
}

}  // namespace gccl
