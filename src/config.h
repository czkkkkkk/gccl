#pragma once

#include <map>
#include <string>
#include <vector>

namespace gccl {

struct Config {
  int world_size;
  int n_blocks;
  int n_threads;         // nthreads per block
  int buffer_size;       // buffer_size in bytes
  int max_feat_size;     // max feature size in 4 bytes
  int threads_per_conn;  // greedy comm
  int repeat_ring;       // ring repeat time
  std::vector<std::vector<int>> rings;
  std::map<int, int> rank_to_dev_id;
  std::string comm_pattern;
  std::string dev_graph_file;
  std::string transport_level;
};

void CheckConfig(const Config& config);
Config DefaultConfig(int world_size);
Config LoadConfig(const std::string& file);
void PrintConfig(const Config& config);
void SetConfigInternal(Config* config, const std::string& config_json);

}  // namespace gccl
