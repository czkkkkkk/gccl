#include "config.h"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>

#include "glog/logging.h"

#include "param.h"
#include "utils.h"

namespace gccl {

using json = nlohmann::json;

void CheckConfig(const Config& config) {
  int world_size = config.world_size;
  int n_blocks = config.n_blocks;
  const auto& rings = config.rings;
  CHECK(config.rings.size() == n_blocks);
  for (int i = 0; i < n_blocks; ++i) {
    CHECK(rings[i].size() == world_size);
    std::vector<bool> vis(world_size, false);
    for (int j = 0; j < world_size; ++j) {
      CHECK(rings[i][j] >= 0 && rings[i][j] < world_size);
      vis[rings[i][j]] = true;
    }
    for (auto v : vis) {
      CHECK(v);
    }
  }
  CHECK(config.buffer_size >= config.n_threads * config.max_feat_size * 4);
}

Config DefaultConfig(int world_size) {
  Config config;
  config.world_size = world_size;
  config.n_blocks = 1;
  config.n_threads = 256;
  config.max_feat_size = 256;
  config.buffer_size = config.n_threads * config.max_feat_size * 4;
  config.rings = {std::vector<int>(world_size)};
  config.comm_pattern = "RING";
  config.transport_level = "IB_NET";
  config.threads_per_conn = 32;
  config.repeat_ring = 1;

  std::iota(config.rings[0].begin(), config.rings[0].end(), 0);
  CheckConfig(config);
  return config;
}

Config LoadConfig(const std::string& file) {
  std::ifstream input(file);
  json config_json;
  input >> config_json;
  int world_size;
  config_json.at("world_size").get_to(world_size);
  Config config = DefaultConfig(world_size);
  config_json.at("n_blocks").get_to(config.n_blocks);
  config_json.at("n_threads").get_to(config.n_threads);
  config_json.at("max_feat_size").get_to(config.max_feat_size);
  int buffer_size_log2;
  config_json.at("buffer_size_log2").get_to(buffer_size_log2);
  config.buffer_size = 1 << buffer_size_log2;
  config_json.at("rings").get_to(config.rings);
  if (config_json.contains("rank_to_dev_id")) {
    config_json.at("rank_to_dev_id").get_to(config.rank_to_dev_id);
  }
  if (config_json.contains("comm_pattern")) {
    config_json.at("comm_pattern").get_to(config.comm_pattern);
  }
  if (config_json.contains("dev_graph_file")) {
    config_json.at("dev_graph_file").get_to(config.dev_graph_file);
  }
  if (config_json.contains("transport_level")) {
    config_json.at("transport_level").get_to(config.transport_level);
  }
  if(config_json.contains("threads_per_conn")) {
    config_json.at("threads_per_conn").get_to(config.threads_per_conn);
  }
  if(config_json.contains("repeat_ring")) {
    config_json.at("repeat_ring").get_to(config.repeat_ring);
  }
  int org_ring_size = config.rings.size();
  config.rings.resize(org_ring_size * config.repeat_ring);
  for(int i = 1; i < config.repeat_ring; ++i) {
    for(int j = 0; j < org_ring_size; ++j) {
      config.rings[i * org_ring_size + j] = config.rings[j];
    }
  }
  if (config.rings.size() > config.n_blocks) {
    config.rings.resize(config.n_blocks);
  }
  std::string comm_pattern = GetEnvParam("COMM_PATTERN", std::string("GREEDY"));
  config.comm_pattern = comm_pattern;
  CheckConfig(config);
  return config;
}

void PrintConfig(const Config& config) {
  LOG(INFO) << "GCCL Config:";
  LOG(INFO) << "  # World size: " << config.world_size;
  LOG(INFO) << "  # N blocks: " << config.n_blocks;
  LOG(INFO) << "  # N threads: " << config.n_threads;
  LOG(INFO) << "  # Max feature size: " << config.max_feat_size;
  LOG(INFO) << "  # Buffer size: " << config.buffer_size;
  LOG(INFO) << "  # Threads per conn: " << config.threads_per_conn;
  for (int i = 0; i < config.n_blocks; ++i) {
    DLOG(INFO) << "    # Ring " << i << " : " << VecToString(config.rings[i]);
  }
  LOG(INFO) << "  # Comm pattern is " << config.comm_pattern;
}

void SetConfigInternal(Config* config, const std::string& config_json) {
  auto j = json::parse(config_json);
  if (j.contains("n_blocks")) {
    j["n_blocks"].get_to(config->n_blocks);
  }
  if (j.contains("rings")) {
    j["rings"].get_to(config->rings);
  }
  if (j.contains("rank_to_dev_id")) {
    std::map<std::string, int> mp;
    j["rank_to_dev_id"].get_to(mp);
    for (auto pair : mp) {
      int rank = std::stoi(pair.first);
      int dev_id = pair.second;
      config->rank_to_dev_id[rank] = dev_id;
      LOG(INFO) << " set dev id of rank " << rank << " to " << dev_id;
    }
  }
  if (j.contains("comm_pattern")) {
    j["comm_pattern"].get_to(config->comm_pattern);
  }

  CheckConfig(*config);
  PrintConfig(*config);
}

}  // namespace gccl
