#include "comm/pattern/all_to_all_comm_pattern.h"

#include "conn/connection.h"
#include "conn/net_connection.h"
#include "gpu/common.h"

namespace gccl {

BinStream &AllToAllCommPatternInfo::serialize(BinStream &stream) const {
  stream << n_peers << rank << buffer_size << max_comm_size << threads_per_conn;
  int send_size = send_off[n_peers];
  int recv_size = recv_off[n_peers];
  stream << std::vector<int>(send_ids, send_ids + send_size);
  stream << std::vector<int>(send_off, send_off + n_peers + 1);
  stream << std::vector<int>(recv_ids, recv_ids + recv_size);
  stream << std::vector<int>(recv_off, recv_off + n_peers + 1);
  return stream;
}

BinStream &AllToAllCommPatternInfo::deserialize(BinStream &stream) {
  stream >> n_peers >> rank >> buffer_size >> max_comm_size >> threads_per_conn;
  std::vector<int> vsend_ids, vsend_off, vrecv_ids, vrecv_off;
  stream >> vsend_ids >> vsend_off >> vrecv_ids >> vrecv_off;
  CopyVectorToRawPtr(&send_ids, vsend_ids);
  CopyVectorToRawPtr(&send_off, vsend_off);
  CopyVectorToRawPtr(&recv_ids, vrecv_ids);
  CopyVectorToRawPtr(&recv_off, vrecv_off);
  return stream;
}

void AllToAllCommPatternInfo::CopyGraphInfoToDev() {
  int send_size = send_off[n_peers];
  int recv_size = recv_off[n_peers];
  GCCLCallocAndCopy(&cpu_send_off,
                    std::vector<int>(send_off, send_off + n_peers + 1));
  GCCLCallocAndCopy(&cpu_recv_off,
                    std::vector<int>(recv_off, recv_off + n_peers + 1));
  GCCLMallocAndCopy(&send_off, send_off, n_peers + 1);
  GCCLMallocAndCopy(&recv_off, recv_off, n_peers + 1);
  GCCLMallocAndCopy(&send_ids, send_ids, send_size);
  GCCLMallocAndCopy(&recv_ids, recv_ids, recv_size);
}
std::vector<CommPatternInfo> AllToAllCommPattern::BuildCommPatternInfos(
    Config *config, const std::vector<std::map<int, int>> &local_mappings,
    const TransferRequest &req, int n_parts) {
  std::vector<CommPatternInfo> ret(n_parts);
  std::vector<AllToAllCommPatternInfo *> aa_infos;
  for (auto &info : ret) {
    info.type = AllToAll;
    aa_infos.push_back(info.GetAllToAllCommPatternInfo());
  }
  for (int i = 0; i < n_parts; ++i) {
    aa_infos[i]->n_peers = n_parts;
    aa_infos[i]->rank = i;
    aa_infos[i]->buffer_size = config->buffer_size;
    aa_infos[i]->threads_per_conn = config->threads_per_conn;
  }

  std::vector<std::vector<int>> send_off(n_parts), recv_off(n_parts),
      send_ids(n_parts), recv_ids(n_parts);
  int max_comm_size = 0;
  for (int i = 0; i < n_parts; ++i) {
    send_off[i].push_back(0);
    recv_off[i].push_back(0);
  }
  for (int i = 0; i < n_parts; ++i) {
    for (int j = 0; j < n_parts; ++j) {
      const auto &ids = req.req_ids[i][j];
      for (auto id : ids) {
        int v = local_mappings[i].at(id);
        send_ids[i].push_back(v);
      }
      int prev = send_off[i].back();
      send_off[i].push_back(send_ids[i].size());
      int size = send_ids[i].size() - prev;
      max_comm_size = std::max(max_comm_size, size);
    }
  }
  for (int j = 0; j < n_parts; ++j) {
    for (int i = 0; i < n_parts; ++i) {
      const auto &ids = req.req_ids[i][j];
      for (auto id : ids) {
        int v = local_mappings[j].at(id);
        recv_ids[j].push_back(v);
      }
      int prev = recv_off[j].back();
      recv_off[j].push_back(recv_ids[j].size());
      int size = recv_ids[j].size() - prev;
      max_comm_size = std::max(max_comm_size, size);
    }
  }
  for (int i = 0; i < n_parts; ++i) {
    CopyVectorToRawPtr(&aa_infos[i]->send_off, send_off[i]);
    CopyVectorToRawPtr(&aa_infos[i]->send_ids, send_ids[i]);
    CopyVectorToRawPtr(&aa_infos[i]->recv_off, recv_off[i]);
    CopyVectorToRawPtr(&aa_infos[i]->recv_ids, recv_ids[i]);
    aa_infos[i]->max_comm_size = max_comm_size;
  }
  return ret;
}

void AllToAllCommPattern::SetupConnection(CommPatternInfo *info,
                                          Coordinator *coor, int bid,
                                          const std::vector<int> &conn_peers) {
  if (coor->GetNPeers() == 1) return;
  auto *aa_info = info->GetAllToAllCommPatternInfo();
  int my_rank = coor->GetRank();
  int n_peers = coor->GetNPeers();

  std::vector<std::shared_ptr<Connection>> recv_conn, send_conn;
  for (int i = 0; i < n_peers; ++i) {
    recv_conn.push_back(ConnectionFactory::GetConnection(
        conn_type_[i], coor->GetPeerInfos()[my_rank], coor->GetPeerInfos()[i],
        bid));
    send_conn.push_back(ConnectionFactory::GetConnection(
        conn_type_[i], coor->GetPeerInfos()[my_rank], coor->GetPeerInfos()[i],
        bid));
  }
  std::vector<RecvDevMem *> recv_dev_mem(n_peers);
  std::vector<SendDevMem *> send_dev_mem(n_peers);
  std::vector<void *> recv_resources(n_peers, nullptr);
  std::vector<void *> send_resources(n_peers, nullptr);
  std::vector<ConnInfo> conn_info(n_peers);

  for (int i = 1; i < n_peers; ++i) {
    int next = (my_rank + i) % n_peers;
    int prev = (my_rank + n_peers - i) % n_peers;
    ExchangeConnInfo recv_ex_info, send_ex_info;

    recv_conn[prev]->RecvSetup(&recv_dev_mem[prev], &recv_resources[prev],
                               aa_info->buffer_size, &conn_info[prev],
                               &recv_ex_info);
    send_conn[next]->SendSetup(&send_dev_mem[next], &send_resources[next],
                               aa_info->buffer_size, &conn_info[next],
                               &send_ex_info);

    auto next_ex_info = coor->RingExchange(next, recv_ex_info);
    send_conn[next]->SendConn(&conn_info[next], send_resources[next],
                              aa_info->buffer_size, &next_ex_info);

    auto prev_ex_info = coor->RingExchange(prev, send_ex_info);
    recv_conn[prev]->RecvConn(&conn_info[prev], recv_resources[prev],
                              &prev_ex_info);
  }
  GCCLMallocAndCopy(&aa_info->recv_dev_mem, recv_dev_mem);
  GCCLCallocAndCopy(&aa_info->recv_resources, recv_resources);
  GCCLMallocAndCopy(&aa_info->send_dev_mem, send_dev_mem);
  GCCLCallocAndCopy(&aa_info->send_resources, send_resources);
  GCCLMallocAndCopy(&aa_info->conn_info, conn_info);

  // Setup proxy here
  auto peer_infos = coor->GetPeerInfos();

  std::string my_hostname = peer_infos[my_rank].hostname;
  std::vector<transportProxyInfo *> send_proxies, recv_proxies;
  for (const auto &p : peer_infos) {
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
  GCCLCallocAndCopy(&aa_info->send_proxy_info, send_proxies);
  GCCLCallocAndCopy(&aa_info->recv_proxy_info, recv_proxies);
}

void AllToAllCommPattern::StartProxy(Coordinator *coor, CommPatternInfo *info) {
  auto *aa_info = info->GetAllToAllCommPatternInfo();
  int n_peers = coor->GetNPeers();
  for (int i = 0; i < n_peers; ++i) {
    transportStartProxy(aa_info->send_proxy_info[i]);
    transportStartProxy(aa_info->recv_proxy_info[i]);
  }
  pthread_yield();  // Let other threads run
}
void AllToAllCommPattern::SaveProxy(Coordinator *coor, CommPatternInfo *info,
                                    int feat_size, int n_threads,
                                    bool forward) {
  auto *aa_info = info->GetAllToAllCommPatternInfo();
  int n_peers = coor->GetNPeers();
  for (int i = 0; i < n_peers; ++i) {
    gcclProxyArgs args;
    args.n_stages = 1;
    args.max_comm_size = {aa_info->max_comm_size};
    args.buffer_size = aa_info->buffer_size;
    args.feat_size = feat_size;
    args.n_threads = n_threads;
    args.resources = aa_info->send_resources[i];
    auto send_comm_off = {aa_info->cpu_send_off[i + 1] -
                          aa_info->cpu_send_off[i]};
    auto recv_comm_off = {aa_info->cpu_recv_off[i + 1] -
                          aa_info->cpu_recv_off[i]};
    if (!forward) {
      std::swap(send_comm_off, recv_comm_off);
    }
    args.comm_off = send_comm_off;
    transportSaveProxy(aa_info->send_proxy_info[i], &args);
    args.resources = aa_info->recv_resources[i];
    args.comm_off = recv_comm_off;
    transportSaveProxy(aa_info->recv_proxy_info[i], &args);
  }
}

}  // namespace gccl
