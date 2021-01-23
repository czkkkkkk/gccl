#pragma once

#include <fstream>

#include "nlohmann/json.hpp"

#include "comm/pattern/comm_pattern.h"
#include "topo/dev_graph.h"
#include "graph.h"

#include "glog/logging.h"


namespace gccl {

using json = nlohmann::json;

inline void from_json(const json &j, TransferRequest &req) {
  j.at("req_ids").get_to(req.req_ids);
}

inline void to_json(json &j, const TransferRequest &req) {
  j["req_ids"] = req.req_ids;
}

inline void from_json(const json& j, Edge& e) {
  j.at("src").get_to(e.u);
  j.at("dst").get_to(e.v);
  std::string type;
  j.at("type").get_to(type);
  e.type = StringToTransportLevel(type);
}

inline void to_json(json& j, const Edge& e) { CHECK(false); }

inline json ReadJsonFromFile(const std::string &file) {
  std::ifstream in(file, std::ios::in);
  CHECK(in.is_open()) << "Cannot open json file " << file;
  json j;
  in >> j;
  return j;
}

inline void WriteJsonToFile(const json &j, const std::string &file) {
  std::ofstream out(file, std::fstream::out);
  CHECK(out.is_open()) << "Cannot open json file " << file;
  out << j;
}

inline void from_json(const json &j, LocalGraphInfo &info) {
  j.at("n_nodes").get_to(info.n_nodes);
  j.at("n_local_nodes").get_to(info.n_local_nodes);
}
inline void to_json(json &j, const LocalGraphInfo &info) {
  j["n_nodes"] = info.n_nodes;
  j["n_local_nodes"] = info.n_local_nodes;
}

}
