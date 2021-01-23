#include "comm/pattern/ring_comm_pattern.h"

#include "conn/net_connection.h"
#include "core.h"
#include "gpu/common.h"
#include "transport.h"
#include "utils.h"

namespace gccl {

BinStream &RingCommPatternInfo::serialize(BinStream &stream) const {
  stream << n_stages << extra_buff_size;
  int send_size = send_off[n_stages];
  int recv_size = recv_off[n_stages];
  stream << std::vector<int>(send_ids, send_ids + send_size);
  stream << std::vector<int>(send_off, send_off + n_stages + 1);
  stream << std::vector<int>(recv_ids, recv_ids + recv_size);
  stream << std::vector<int>(recv_off, recv_off + n_stages + 1);
  stream << std::vector<int>(max_comm_size, max_comm_size + n_stages);
  stream << buffer_size << max_feat_size;
  return stream;
}

BinStream &RingCommPatternInfo::deserialize(BinStream &stream) {
  stream >> n_stages >> extra_buff_size;
  std::vector<int> vsend_ids, vsend_off, vrecv_ids, vrecv_off, vmax_comm_size;
  stream >> vsend_ids >> vsend_off >> vrecv_ids >> vrecv_off >> vmax_comm_size;
  stream >> buffer_size >> max_feat_size;

  CopyVectorToRawPtr(&send_ids, vsend_ids);
  CopyVectorToRawPtr(&send_off, vsend_off);
  CopyVectorToRawPtr(&recv_ids, vrecv_ids);
  CopyVectorToRawPtr(&recv_off, vrecv_off);
  CopyVectorToRawPtr(&max_comm_size, vmax_comm_size);
  return stream;
}

void RingCommPatternInfo::CopyGraphInfoToDev() {
  int send_size = send_off[n_stages];
  int recv_size = recv_off[n_stages];
  GCCLCallocAndCopy(&cpu_max_comm_size,
                    std::vector<int>(max_comm_size, max_comm_size + n_stages));
  GCCLCallocAndCopy(&cpu_send_off,
                    std::vector<int>(send_off, send_off + n_stages + 1));
  GCCLCallocAndCopy(&cpu_recv_off,
                    std::vector<int>(recv_off, recv_off + n_stages + 1));
  GCCLMallocAndCopy(&send_off, send_off, n_stages + 1);
  GCCLMallocAndCopy(&recv_off, recv_off, n_stages + 1);
  GCCLMallocAndCopy(&send_ids, send_ids, send_size);
  GCCLMallocAndCopy(&recv_ids, recv_ids, recv_size);
  GCCLMallocAndCopy(&max_comm_size, max_comm_size, n_stages);
}

void RingCommPatternInfo::Print() const {
  LOG(INFO) << "  Number of stages " << n_stages;
  LOG(INFO) << "  Extra buff size " << extra_buff_size;
  LOG(INFO) << "  Max comm size " << VecToString(std::vector<int>(max_comm_size, max_comm_size + n_stages));
  for (int j = 0; j < n_stages; ++j) {
    LOG(INFO) << "    Send size of " << j << " stage is "
              << send_off[j + 1] - send_off[j];
  }
}

RingTransferInfo BuildRingTransferInfo(const TransferRequest &req, int n_parts,
                                       const std::vector<int> &dev_topo,
                                       const std::vector<int> &dev_topo_rmap) {
  RingTransferInfo ret({Build3DVector<int>(n_parts, n_parts - 1)});
  for (int i = 0; i < n_parts; ++i) {
    for (int j = 0; j < n_parts; ++j) {
      if (i == j) continue;
      for (auto u : req.req_ids[i][j]) {
        int ti = i;
        int step = 0;
        while (ti != j) {
          ret.tr_ids[ti][step].push_back(u);
          step++;
          int r_pos = dev_topo_rmap[ti];
          r_pos = (r_pos + 1) % n_parts;
          ti = dev_topo[r_pos];
        }
      }
    }
  }
  for (auto &vecs : ret.tr_ids) {
    for (auto &vec : vecs) {
      UniqueVec(vec);
    }
  }
  return ret;
}

void RingCommPattern::SortTransferInfoByLocalId(
    RingTransferInfo &tr_info,
    const std::vector<std::map<int, int>> &local_mappings, int n_parts) {
  for (int i = 0; i < n_parts; ++i) {
    int next = dev_topo_rmap_[i];
    next = (next + 1) % n_parts;
    next = dev_topo_[next];
    for (auto &vec : tr_info.tr_ids[i]) {
      std::sort(vec.begin(), vec.end(), [&local_mappings, next](int l, int r) {
        bool has_l = local_mappings[next].count(l) > 0;
        bool has_r = local_mappings[next].count(r) > 0;
        if (has_l != has_r) {
          return has_l > has_r;
        }
        if (!has_l) return l < r;
        return local_mappings[next].at(l) < local_mappings[next].at(r);
      });
    }
  }
}

std::vector<CommPatternInfo> RingCommPattern::BuildCommPatternInfos(
    Config *config, const std::vector<std::map<int, int>> &local_mappings,
    const TransferRequest &req, int n_parts) {
  auto pattern_infos = std::vector<CommPatternInfo>(n_parts);
  auto ring_infos = std::vector<RingCommPatternInfo *>(n_parts);
  static_assert(sizeof(RingCommPatternInfo) <= MAX_COMM_PATTERN_INFO_SIZE, "");
  for (int i = 0; i < n_parts; ++i) {
    pattern_infos[i].type = CommPatternType::Ring;
    ring_infos[i] = (RingCommPatternInfo *)pattern_infos[i].data;
  }

  auto transfer_infos =
      BuildRingTransferInfo(req, n_parts, dev_topo_, dev_topo_rmap_);
  SortTransferInfoByLocalId(transfer_infos, local_mappings, n_parts);
  // n_parts - 1 stages
  std::vector<std::vector<int>> send_ids(n_parts), send_off(n_parts),
      recv_ids(n_parts), recv_off(n_parts);
  std::vector<std::map<int, int>> extra_node_to_buff_index(n_parts);
  for (int i = 0; i < n_parts; ++i) {
    ring_infos[i]->extra_buff_size = 1;
    ring_infos[i]->buffer_size = config->buffer_size;
    ring_infos[i]->max_feat_size = config->max_feat_size;
  }
  std::vector<int> max_comm_size;

  for (int i = 0; i < n_parts; ++i) {
    send_off[i].push_back(0);
    recv_off[i].push_back(0);
  }
  // Map global node id to local node id
  for (int stage = 0; stage < n_parts - 1; ++stage) {
    int max_comm = 0;
    for (int i = 0; i < n_parts; ++i) {
      for (auto u : transfer_infos.tr_ids[i][stage]) {
        if (local_mappings[i].count(u) > 0) {
          send_ids[i].push_back(local_mappings[i].at(u));
        } else {
          CHECK(extra_node_to_buff_index[i].count(u) > 0)
              << "Device " << i << " do not have " << u << " at stage "
              << stage;
          send_ids[i].push_back(ENCODE(extra_node_to_buff_index[i].at(u)));
        }
      }
      send_off[i].push_back(send_ids[i].size());
      max_comm = std::max((int)(send_off[i][stage + 1] - send_off[i][stage]),
                          max_comm);

      // Using different buffer for send and receive
      extra_node_to_buff_index[i].clear();

      int pre = dev_topo_rmap_[i];
      pre = (pre + n_parts - 1) % n_parts;
      pre = dev_topo_[pre];
      for (auto u : transfer_infos.tr_ids[pre][stage]) {
        if (local_mappings[i].count(u) > 0) {
          recv_ids[i].push_back(local_mappings[i].at(u));
        } else {
          int sz = extra_node_to_buff_index[i].size();
          recv_ids[i].push_back(ENCODE(sz));
          extra_node_to_buff_index[i][u] = sz++;
        }
      }
      recv_off[i].push_back(recv_ids[i].size());

      max_comm = std::max((int)(recv_off[i][stage + 1] - recv_off[i][stage]),
                          max_comm);

      ring_infos[i]->extra_buff_size =
          std::max(ring_infos[i]->extra_buff_size,
                   (int)extra_node_to_buff_index[i].size());
    }
    max_comm_size.push_back(max_comm);
  }

  for (int i = 0; i < n_parts; ++i) {
    CopyVectorToRawPtr(&ring_infos[i]->send_ids, send_ids[i]);
    CopyVectorToRawPtr(&ring_infos[i]->send_off, send_off[i]);
    CopyVectorToRawPtr(&ring_infos[i]->recv_ids, recv_ids[i]);
    CopyVectorToRawPtr(&ring_infos[i]->recv_off, recv_off[i]);
    CopyVectorToRawPtr(&ring_infos[i]->max_comm_size, max_comm_size);
    ring_infos[i]->n_stages = n_parts - 1;
  }
  return pattern_infos;
}

void BuildRingConnection(Coordinator *coor, int prev, int next,
                         std::shared_ptr<Connection> recv_conn,
                         std::shared_ptr<Connection> send_conn,
                         RingConn *ring_conn, int buffer_size) {
  ExchangeConnInfo recv_ex_info, send_ex_info;

  recv_conn->RecvSetup(&ring_conn->recv_dev_mem, &ring_conn->recv_resources,
                       buffer_size, &ring_conn->conn_info, &recv_ex_info);
  send_conn->SendSetup(&ring_conn->send_dev_mem, &ring_conn->send_resources,
                       buffer_size, &ring_conn->conn_info, &send_ex_info);

  auto next_ex_info = coor->RingExchange(next, recv_ex_info);
  send_conn->SendConn(&ring_conn->conn_info, ring_conn->send_resources,
                      buffer_size, &next_ex_info);

  auto prev_ex_info = coor->RingExchange(prev, send_ex_info);
  recv_conn->RecvConn(&ring_conn->conn_info, ring_conn->recv_resources,
                      &prev_ex_info);
}

void RingCommPattern::SetupConnection(CommPatternInfo *info, Coordinator *coor,
                                      int bid,
                                      const std::vector<int> &conn_peer) {
  if (coor->GetNPeers() == 1) return;

  auto *ring_info = (RingCommPatternInfo *)info->data;

  int my_rank = coor->GetRank();
  int n_peers = coor->GetNPeers();

  // setup extra buffer
  int max_feat_size = ring_info->max_feat_size;
  int extra_mem_size = ring_info->extra_buff_size * max_feat_size * 4;
  extra_mem_size = std::max(extra_mem_size, 1);
  GCCLCudaMalloc((char **)&ring_info->dev_extra_mem, extra_mem_size);

  int prev = (dev_topo_rmap_[my_rank] + n_peers - 1) % n_peers;
  prev = dev_topo_[prev];
  int next = (dev_topo_rmap_[my_rank] + 1) % n_peers;
  next = dev_topo_[next];

  auto prev_conn = ConnectionFactory::GetConnection(
      conn_type_[prev], coor->GetPeerInfos()[my_rank],
      coor->GetPeerInfos()[prev], bid);
  auto next_conn = ConnectionFactory::GetConnection(
      conn_type_[next], coor->GetPeerInfos()[my_rank],
      coor->GetPeerInfos()[next], bid);
  BuildRingConnection(coor, prev, next, prev_conn, next_conn,
                      &ring_info->forward_conn, ring_info->buffer_size);
  BuildRingConnection(coor, next, prev, next_conn, prev_conn,
                      &ring_info->backward_conn, ring_info->buffer_size);
  // Setup proxy
  auto peer_infos = coor->GetPeerInfos();

  std::string my_hostname = peer_infos[my_rank].hostname;
  transportProxyInfo *fw_send_proxy, *fw_recv_proxy;
  transportProxyInfo *bw_send_proxy, *bw_recv_proxy;
  if (peer_infos[next].hostname == my_hostname) {
    fw_send_proxy = nullptr;
    bw_recv_proxy = nullptr;
  } else {
    CreateProxy(&fw_send_proxy, &NetSendProxy);
    CreateProxy(&bw_recv_proxy, &NetRecvProxy);
  }
  if (peer_infos[prev].hostname == my_hostname) {
    fw_recv_proxy = nullptr;
    bw_send_proxy = nullptr;
  } else {
    CreateProxy(&fw_recv_proxy, &NetRecvProxy);
    CreateProxy(&bw_send_proxy, &NetSendProxy);
  }
  ring_info->forward_conn.send_proxy = fw_send_proxy;
  ring_info->forward_conn.recv_proxy = fw_recv_proxy;
  ring_info->backward_conn.send_proxy = bw_send_proxy;
  ring_info->backward_conn.recv_proxy = bw_recv_proxy;
}

void RingCommPattern::StartProxy(Coordinator *coor, CommPatternInfo *raw_info) {
  auto *info = raw_info->GetRingCommPatternInfo();
  transportStartProxy(info->forward_conn.send_proxy);
  transportStartProxy(info->forward_conn.recv_proxy);
  transportStartProxy(info->backward_conn.send_proxy);
  transportStartProxy(info->backward_conn.recv_proxy);
  pthread_yield();  // Let other threads run
}
void SaveProxyOnConn(transportProxyInfo *proxy, int n_stages, int buffer_size,
                     int feat_size, int n_threads, void *resources,
                     const std::vector<int> &max_comm_size,
                     const std::vector<int> &comm_off) {
  gcclProxyArgs args;
  args.n_stages = n_stages;
  args.buffer_size = buffer_size;
  args.feat_size = feat_size;
  args.n_threads = n_threads;
  args.resources = resources;
  args.max_comm_size = max_comm_size;
  args.comm_off = comm_off;
  transportSaveProxy(proxy, &args);
}

void RingCommPattern::SaveProxy(Coordinator *coor, CommPatternInfo *raw_info,
                                int feat_size, int n_threads, bool forward) {
  auto *info = raw_info->GetRingCommPatternInfo();
  int n_peers = coor->GetNPeers();
  int n_stages = info->n_stages;
  auto max_comm_size = std::vector<int>(info->cpu_max_comm_size,
                                        info->cpu_max_comm_size + n_stages);
  std::vector<int> send_comm_off, recv_comm_off;
  for (int i = 0; i < n_stages; ++i) {
    auto *send_ptr = info->cpu_send_off;
    auto *recv_ptr = info->cpu_recv_off;
    send_comm_off.push_back(send_ptr[i + 1] - send_ptr[i]);
    recv_comm_off.push_back(recv_ptr[i + 1] - recv_ptr[i]);
  }
  if (!forward) {
    std::reverse(max_comm_size.begin(), max_comm_size.end());
    std::swap(send_comm_off, recv_comm_off);
    std::reverse(send_comm_off.begin(), send_comm_off.end());
    std::reverse(recv_comm_off.begin(), recv_comm_off.end());
  }
  if (forward) {
    SaveProxyOnConn(info->forward_conn.send_proxy, n_stages, info->buffer_size,
                    feat_size, n_threads, info->forward_conn.send_resources, max_comm_size,
                    send_comm_off);
    SaveProxyOnConn(info->forward_conn.recv_proxy, n_stages, info->buffer_size,
                    feat_size, n_threads, info->forward_conn.recv_resources, max_comm_size,
                    recv_comm_off);
  } else {
    SaveProxyOnConn(info->backward_conn.send_proxy, n_stages, info->buffer_size,
                    feat_size, n_threads, info->backward_conn.send_resources,
                    max_comm_size, send_comm_off);
    SaveProxyOnConn(info->backward_conn.recv_proxy, n_stages, info->buffer_size,
                    feat_size, n_threads, info->backward_conn.recv_resources,
                    max_comm_size, recv_comm_off);
  }
}

}  // namespace gccl
