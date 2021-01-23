#include "comm/pattern/greedy_comm_pattern.h"

#include <map>
#include <sstream>
#include <iomanip>

#include "comm/pattern/cost_model.h"
#include "comm/pattern/greedy_algo.h"
#include "conn/net_connection.h"
#include "core.h"
#include "gpu/common.h"
#include "transport.h"
#include "utils.h"

namespace gccl {

struct NodeReq {
  int src_dev, node_id;
  std::vector<int> devs;
};

BinStream &GreedyCommPatternInfo::serialize(BinStream &stream) const {
  stream << n_peers << rank << n_stages << extra_buffer_size << buffer_size
         << max_feat_size << threads_per_conn;
  int send_size = send_off[n_stages * n_peers];
  int recv_size = recv_off[n_stages * n_peers];
  stream << std::vector<int>(send_ids, send_ids + send_size);
  stream << std::vector<int>(send_off, send_off + n_stages * n_peers + 1);
  stream << std::vector<int>(recv_ids, recv_ids + recv_size);
  stream << std::vector<int>(recv_off, recv_off + n_stages * n_peers + 1);
  stream << std::vector<int>(max_comm_size, max_comm_size + n_stages);
  return stream;
}
BinStream &GreedyCommPatternInfo::deserialize(BinStream &stream) {
  stream >> n_peers >> rank >> n_stages >> extra_buffer_size >> buffer_size >>
      max_feat_size >> threads_per_conn;
  std::vector<int> vsend_ids, vsend_off, vrecv_ids, vrecv_off, vmax_comm_size;
  stream >> vsend_ids >> vsend_off >> vrecv_ids >> vrecv_off >> vmax_comm_size;
  CopyVectorToRawPtr(&send_ids, vsend_ids);
  CopyVectorToRawPtr(&send_off, vsend_off);
  CopyVectorToRawPtr(&recv_ids, vrecv_ids);
  CopyVectorToRawPtr(&recv_off, vrecv_off);
  CopyVectorToRawPtr(&max_comm_size, vmax_comm_size);
  return stream;
}

void GreedyCommPatternInfo::CopyGraphInfoToDev() {
  int send_size = send_off[n_stages * n_peers];
  int recv_size = recv_off[n_stages * n_peers];
  GCCLCallocAndCopy(&cpu_max_comm_size,
                    std::vector<int>(max_comm_size, max_comm_size + n_stages));
  GCCLCallocAndCopy(
      &cpu_send_off,
      std::vector<int>(send_off, send_off + n_stages * n_peers + 1));
  GCCLCallocAndCopy(
      &cpu_recv_off,
      std::vector<int>(recv_off, recv_off + n_stages * n_peers + 1));
  GCCLMallocAndCopy(&send_off, send_off, n_stages * n_peers + 1);
  GCCLMallocAndCopy(&recv_off, recv_off, n_stages * n_peers + 1);
  GCCLMallocAndCopy(&send_ids, send_ids, send_size);
  GCCLMallocAndCopy(&recv_ids, recv_ids, recv_size);
  GCCLMallocAndCopy(&max_comm_size, max_comm_size, n_stages);
}
void GreedyCommPatternInfo::Print() const {
  DLOG(INFO) << "  Number of stages " << n_stages;
  DLOG(INFO) << "  Extra buff size " << extra_buffer_size;
  std::vector<std::vector<int>> send(n_stages, std::vector<int>(n_peers));
  std::vector<std::vector<int>> recv(n_stages, std::vector<int>(n_peers));
  for(int i = 0; i < n_stages; ++i) {
    for(int j = 0; j < n_peers; ++j) {
      int pos = n_peers * i + j;
      send[i][j] = send_off[pos + 1] - send_off[pos];
      recv[i][j] = recv_off[pos + 1] - recv_off[pos];
    }
  }
  {
    std::stringstream ss;
    ss << "Send size table\n";
    for(int i = 0; i < n_stages; ++i) {
      ss << "stages " << i << ":";
      for(int j = 0; j < n_peers; ++j) {
        ss << std::fixed << std::setfill(' ') << std::setw(8) << send[i][j];
      }
      ss << '\n';
    }
    DLOG(INFO) << ss.str();
  }
  {
    std::stringstream ss;
    ss << "Recv size table\n";
    for(int i = 0; i < n_stages; ++i) {
      ss << "stages " << i << ":";
      for(int j = 0; j < n_peers; ++j) {
        ss << std::fixed << std::setfill(' ') << std::setw(8) << recv[i][j];
      }
      ss << '\n';
    }
    DLOG(INFO) << ss.str();
  }
}

int GreedyCommPatternInfo::GetMemBytes() const {
  int ret = 0;
  int send_size = send_off[n_stages * n_peers];
  int recv_size = recv_off[n_stages * n_peers];
  return (send_size + recv_size) * 4;
}

std::vector<NodeReq> BuildNodeToReqDevsMaps(const TransferRequest &req,
                                            int n_parts) {
  std::vector<NodeReq> ret;
  for (int i = 0; i < n_parts; ++i) {
    std::vector<std::pair<int, int>> node_to_peers;
    for (int j = 0; j < n_parts; ++j) {
      for (auto v : req.req_ids[i][j]) {
        node_to_peers.push_back({v, j});
      }
    }
    std::sort(node_to_peers.begin(), node_to_peers.end());
    for (int j = 0, k = 0; j < node_to_peers.size(); j = k) {
      int u = node_to_peers[j].first;
      std::vector<int> devs;
      devs.push_back(i);
      while (k < node_to_peers.size() && node_to_peers[k].first == u) {
        devs.push_back(node_to_peers[k].second);
        ++k;
      }
      ret.push_back({i, u, devs});
    }
  }
  return ret;
}

GreedyTransferInfo BuildGreedyTransferInfo(CostModel *cost_model,
                                           const TransferRequest &req,
                                           int n_parts) {
  GreedyTransferInfo ret;
  ret.tr_ids = std::vector<Vec3D<int>>(
      n_parts, Build3DVector<int>(n_parts, n_parts - 1));
  auto node_to_req_devs = BuildNodeToReqDevsMaps(req, n_parts);
  // Random shuffle to avoid constant pattern
  std::vector<int> shuffle_ids(node_to_req_devs.size());
  std::iota(shuffle_ids.begin(), shuffle_ids.end(), 0);
  std::random_shuffle(shuffle_ids.begin(), shuffle_ids.end());
  for (int id : shuffle_ids) {
    const auto &req = node_to_req_devs[id];
    int v = req.node_id;
    auto decision = GreedyAlgo::MakeDecision(cost_model, req.devs, n_parts);
    for (int i = 0; i < decision.n_stages; ++i) {
      int l = decision.offset[i];
      int r = decision.offset[i + 1];
      for (int j = l; j < r; ++j) {
        int src = decision.send_pairs[j].first;
        int dst = decision.send_pairs[j].second;
        ret.tr_ids[src][dst][i].push_back(v);
      }
    }
  }
  return ret;
}

int GetMaxStage(const GreedyTransferInfo &info, int n_parts) {
  int stage = 0;
  while (stage < n_parts - 1) {
    bool found = false;
    for (int i = 0; i < n_parts; ++i) {
      for (int j = 0; j < n_parts; ++j) {
        if (info.tr_ids[i][j][stage].size() > 0) {
          found = true;
        }
      }
      if (found) break;
    }
    if (!found) break;
    stage++;
  }
  return stage;
}

std::vector<CommPatternInfo> GreedyCommPattern::BuildCommPatternInfos(
    Config *config, const std::vector<std::map<int, int>> &local_mappings,
    const TransferRequest &req, int n_parts) {
  std::vector<CommPatternInfo> ret(n_parts);
  std::vector<GreedyCommPatternInfo *> infos;
  for (auto &info : ret) {
    info.type = Greedy;
    infos.push_back(info.GetGreedyCommPatternInfo());
  }
  std::shared_ptr<DevGraph> dev_graph;
  if (config->dev_graph_file.size() == 0) {
    dev_graph = std::make_shared<DevGraph>(DefaultDevGraph(n_parts));
  } else {
    dev_graph = std::make_shared<DevGraph>(config->dev_graph_file);
  }
  CostModel cost_model(dev_graph.get(),
                       StringToTransportLevel(config->transport_level),
                       n_parts);
  for (int i = 0; i < n_parts; ++i) {
    infos[i]->n_peers = n_parts;
    infos[i]->rank = i;
    infos[i]->buffer_size = config->buffer_size;
    infos[i]->max_feat_size = config->max_feat_size;
    infos[i]->extra_buffer_size = 1;
    infos[i]->threads_per_conn = config->threads_per_conn;
  }
  auto tr_info = BuildGreedyTransferInfo(&cost_model, req, n_parts);
  int n_stages = GetMaxStage(tr_info, n_parts);
  // to local ids
  std::vector<std::vector<int>> send_ids(n_parts), send_off(n_parts),
      recv_ids(n_parts), recv_off(n_parts);
  std::vector<int> max_comm_size;
  for (int i = 0; i < n_parts; ++i) {
    send_off[i].push_back(0);
    recv_off[i].push_back(0);
  }
  std::vector<std::map<int, int>> extra_node_to_buff_index(n_parts);
  for (int stage = 0; stage < n_stages; ++stage) {
    int max_comm = 0;
    for (int i = 0; i < n_parts; ++i) {
      for (int j = 0; j < n_parts; ++j) {
        if (i == j) {
          send_off[i].push_back(send_ids[i].size());
          continue;
        }
        for (auto v : tr_info.tr_ids[i][j][stage]) {
          if (local_mappings[i].count(v) > 0) {
            send_ids[i].push_back(local_mappings[i].at(v));
          } else {
            CHECK(extra_node_to_buff_index[i].count(v) > 0);
            send_ids[i].push_back(ENCODE(extra_node_to_buff_index[i].at(v)));
          }
        }
        max_comm =
            std::max(max_comm, (int)send_ids[i].size() - send_off[i].back());
        send_off[i].push_back(send_ids[i].size());
      }  // j for loop
      extra_node_to_buff_index[i].clear();

      for (int j = 0; j < n_parts; ++j) {
        if (i == j) {
          recv_off[i].push_back(recv_ids[i].size());
          continue;
        }
        for (auto v : tr_info.tr_ids[j][i][stage]) {
          if (local_mappings[i].count(v) > 0) {
            recv_ids[i].push_back(local_mappings[i].at(v));
          } else {
            CHECK(extra_node_to_buff_index[i].count(v) == 0);
            int sz = extra_node_to_buff_index[i].size();
            recv_ids[i].push_back(ENCODE(sz));
            extra_node_to_buff_index[i][v] = sz++;
          }
        }
        max_comm =
            std::max(max_comm, (int)recv_ids[i].size() - recv_off[i].back());
        recv_off[i].push_back(recv_ids[i].size());
      }  // j for loop
      infos[i]->extra_buffer_size = std::max(
          infos[i]->extra_buffer_size, (int)extra_node_to_buff_index[i].size());
    }  // i for loop
    max_comm_size.push_back(max_comm);
  }  // stage for loop
  for (int i = 0; i < n_parts; ++i) {
    CopyVectorToRawPtr(&infos[i]->send_ids, send_ids[i]);
    CopyVectorToRawPtr(&infos[i]->send_off, send_off[i]);
    CopyVectorToRawPtr(&infos[i]->recv_ids, recv_ids[i]);
    CopyVectorToRawPtr(&infos[i]->recv_off, recv_off[i]);
    CopyVectorToRawPtr(&infos[i]->max_comm_size, max_comm_size);
    infos[i]->n_stages = n_stages;
  }
  return ret;
}

std::vector<bool> GetConnMap(GreedyCommPatternInfo *info, int my_rank,
                             int n_peers) {
  std::vector<bool> ret(n_peers, false);
  for (int stage = 0; stage < info->n_stages; ++stage) {
    for (int j = 0; j < n_peers; ++j) {
      if (j == my_rank) continue;
      int t = j;
      int pos = stage * n_peers + t;
      int send_ptr = info->send_off[pos];
      int send_size = info->send_off[pos + 1] - send_ptr;
      if (send_size > 0) ret[j] = true;
      int recv_ptr = info->recv_off[pos];
      int recv_size = info->recv_off[pos + 1] - recv_ptr;
      if (recv_size > 0) ret[j] = true;
    }
  }
  return ret;
}

void GreedyCommPattern::SetupConnection(CommPatternInfo *raw_info,
                                        Coordinator *coor, int bid,
                                        const std::vector<int> &conn_peers) {
  if (coor->GetNPeers() == 1) return;
  auto *info = raw_info->GetGreedyCommPatternInfo();
  int my_rank = coor->GetRank();
  int n_peers = coor->GetNPeers();
  conn_peers_ = conn_peers;
  std::vector<bool> conn_map(n_peers, false);
  for (auto v : conn_peers) {
    conn_map[v] = true;
  }
  DLOG(INFO) << "Connect peers are " << VecToString(conn_peers);

  GCCLMallocAndCopy(&info->conn_peers, conn_peers);
  info->n_conn = conn_peers.size();

  // setup extra buffer
  int max_feat_size = info->max_feat_size;
  int extra_mem_size = info->extra_buffer_size * max_feat_size * 4;
  extra_mem_size = std::max(extra_mem_size, 1);
  GCCLCudaMalloc((char **)&info->dev_extra_mem, extra_mem_size);

  std::vector<std::shared_ptr<Connection>> recv_conn, send_conn;
  for (int i = 0; i < n_peers; ++i) {
    if (conn_map[i]) {
      recv_conn.push_back(ConnectionFactory::GetConnection(
          conn_type_[i], coor->GetPeerInfos()[my_rank], coor->GetPeerInfos()[i],
          bid));
      send_conn.push_back(ConnectionFactory::GetConnection(
          conn_type_[i], coor->GetPeerInfos()[my_rank], coor->GetPeerInfos()[i],
          bid));
    } else {
      recv_conn.push_back(nullptr);
      send_conn.push_back(nullptr);
    }
  }
  std::vector<RecvDevMem *> recv_dev_mem(n_peers, nullptr);
  std::vector<SendDevMem *> send_dev_mem(n_peers, nullptr);
  std::vector<void *> recv_resources(n_peers, nullptr);
  std::vector<void *> send_resources(n_peers, nullptr);
  std::vector<ConnInfo> conn_info(n_peers);

  for (int i = 1; i < n_peers; ++i) {
    int next = (my_rank + i) % n_peers;
    int prev = (my_rank + n_peers - i) % n_peers;
    ExchangeConnInfo recv_ex_info, send_ex_info;

    if (conn_map[prev]) {
      recv_conn[prev]->RecvSetup(&recv_dev_mem[prev], &recv_resources[prev],
                                 info->buffer_size, &conn_info[prev],
                                 &recv_ex_info);
    }
    if (conn_map[next]) {
      send_conn[next]->SendSetup(&send_dev_mem[next], &send_resources[next],
                                 info->buffer_size, &conn_info[next],
                                 &send_ex_info);
    }

    auto next_ex_info = coor->RingExchange(next, recv_ex_info);
    if (conn_map[next]) {
      send_conn[next]->SendConn(&conn_info[next], send_resources[next],
                                info->buffer_size, &next_ex_info);
    }

    auto prev_ex_info = coor->RingExchange(prev, send_ex_info);
    if (conn_map[prev]) {
      recv_conn[prev]->RecvConn(&conn_info[prev], recv_resources[prev],
                                &prev_ex_info);
    }
  }
  GCCLMallocAndCopy(&info->recv_dev_mem, recv_dev_mem);
  GCCLCallocAndCopy(&info->recv_resources, recv_resources);
  GCCLMallocAndCopy(&info->send_dev_mem, send_dev_mem);
  GCCLCallocAndCopy(&info->send_resources, send_resources);
  GCCLMallocAndCopy(&info->conn_info, conn_info);

  // Setup proxy
  auto peer_infos = coor->GetPeerInfos();

  std::string my_hostname = peer_infos[my_rank].hostname;
  std::vector<transportProxyInfo *> send_proxies, recv_proxies;
  for (int i = 0; i < conn_peers.size(); ++i) {
    const auto &p = peer_infos[conn_peers[i]];
    const auto &h = p.hostname;
    if (h == my_hostname) {
      send_proxies.push_back(nullptr);
      recv_proxies.push_back(nullptr);
      continue;
    }
    transportProxyInfo *send_proxy, *recv_proxy;
    // Start send proxy
    // Start recv proxy
    CreateProxy(&send_proxy, &NetSendProxy);
    CreateProxy(&recv_proxy, &NetRecvProxy);
    send_proxies.push_back(send_proxy);
    recv_proxies.push_back(recv_proxy);
  }
  GCCLCallocAndCopy(&info->send_proxy_info, send_proxies);
  GCCLCallocAndCopy(&info->recv_proxy_info, recv_proxies);
  DLOG(INFO) << "Connection setup done for block " << bid;
}
void GreedyCommPattern::StartProxy(Coordinator *coor,
                                   CommPatternInfo *raw_info) {
  auto *info = raw_info->GetGreedyCommPatternInfo();
  for (int i = 0; i < conn_peers_.size(); ++i) {
    transportStartProxy(info->send_proxy_info[i]);
    transportStartProxy(info->recv_proxy_info[i]);
  }
  pthread_yield();  // Let other threads run
}
void GreedyCommPattern::SaveProxy(Coordinator *coor, CommPatternInfo *raw_info,
                                  int feat_size, int n_threads, bool forward) {
  auto *info = raw_info->GetGreedyCommPatternInfo();
  int n_peers = coor->GetNPeers();
  int n_stages = info->n_stages;
  auto max_comm_size = std::vector<int>(info->cpu_max_comm_size,
                                        info->cpu_max_comm_size + n_stages);
  if (!forward) {
    std::reverse(max_comm_size.begin(), max_comm_size.end());
  }
  for (int i = 0; i < conn_peers_.size(); ++i) {
    int peer = conn_peers_[i];

    gcclProxyArgs args;
    args.n_stages = n_stages;
    args.max_comm_size = max_comm_size;
    args.buffer_size = info->buffer_size;
    args.feat_size = feat_size;
    args.n_threads = n_threads;
    std::vector<int> send_comm_off, recv_comm_off;
    for (int stage = 0; stage < n_stages; ++stage) {
      int pos = stage * n_peers + peer;
      int *ptr = info->cpu_send_off;
      send_comm_off.push_back(ptr[pos + 1] - ptr[pos]);
    }
    for (int stage = 0; stage < n_stages; ++stage) {
      int pos = stage * n_peers + peer;
      int *ptr = info->cpu_recv_off;
      recv_comm_off.push_back(ptr[pos + 1] - ptr[pos]);
    }
    if (!forward) {
      std::swap(send_comm_off, recv_comm_off);
      std::reverse(send_comm_off.begin(), send_comm_off.end());
      std::reverse(recv_comm_off.begin(), recv_comm_off.end());
    }
    args.comm_off = send_comm_off;
    args.resources = info->send_resources[peer];
    transportSaveProxy(info->send_proxy_info[i], &args);
    args.comm_off = recv_comm_off;
    args.resources = info->recv_resources[peer];
    transportSaveProxy(info->recv_proxy_info[i], &args);
  }
}

}  // namespace gccl
