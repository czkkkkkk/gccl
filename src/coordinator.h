#pragma once

#include <memory>
#include <vector>

#include "glog/logging.h"
#include "zmq.hpp"

#include "base/bin_stream.h"
#include "gccl.h"

namespace gccl {

gcclUniqueId StringToUniqueId(const std::string &str);
std::string UniqueIdToString(const gcclUniqueId &id);

enum CommEvent { Allgather = 0, Scatter = 1, Broadcast = 2, RingExchange = 3 };

struct ProcInfo {
  int pid;
  int dev_id;
  int rank;
  std::string hostname;
  BinStream &serialize(BinStream &bs) const {
    bs << pid << dev_id << rank << hostname;
    return bs;
  }
  BinStream &deserialize(BinStream &bs) {
    bs >> pid >> dev_id >> rank >> hostname;
    return bs;
  }
};

class Coordinator {
 public:
  Coordinator(zmq::context_t *zmq_ctx);

  void SetRankAndNPeers(int rank, int world_size);

  int GetNPeers() const { return n_peers_; }
  int GetRank() const { return rank_; }
  int GetDevId() const { return peer_infos_[rank_].dev_id; }
  bool IsRoot() const { return is_root_; }
  const std::string &GetHostname() const { return peer_infos_[rank_].hostname; }

  void RootInit(const gcclUniqueId &id);  // For root
  void BuildPeerInfo(const gcclUniqueId &id);
  const std::vector<ProcInfo> &GetPeerInfos() const { return peer_infos_; }

  void ConnectToRoot(const gcclUniqueId &id);
  void SendIntTo(int peer_id, int val);
  void SendBinstreamTo(
      int peer_id,
      const std::shared_ptr<BinStream> &payload);  // -1 means send to root

  int RecvIntFromRoot();
  std::shared_ptr<BinStream> RecvBinstreamFromRoot();

  int RootRecvInt();                               // For root
  std::shared_ptr<BinStream> RootRecvBinStream();  // For root

  void Barrier();

  template <typename T>
  void Allgather(std::vector<T> &vec) {
    CHECK(vec.size() == n_peers_);
    auto bs = std::make_shared<BinStream>();
    *bs << CommEvent::Allgather << rank_ << vec[rank_];
    SendBinstreamTo(-1, bs);
    if (is_root_) {
      std::vector<T> root_vals(n_peers_);
      for (int i = 0; i < n_peers_; ++i) {
        auto recv_bs = RootRecvBinStream();
        CommEvent event;
        int rank;
        T val;
        *recv_bs >> event >> rank >> val;
        CHECK(event == CommEvent::Allgather);
        root_vals[rank] = val;
      }
      auto broadcast_bs = std::make_shared<BinStream>();
      *broadcast_bs << root_vals;
      for (int i = 0; i < n_peers_; ++i) {
        SendBinstreamTo(i, broadcast_bs);
      }
    }
    auto allinfos = RecvBinstreamFromRoot();
    *allinfos >> vec;
  }
  template <typename T>
  T Scatter(const std::vector<T> &vec) {
    if (is_root_) {
      CHECK(vec.size() == n_peers_);
      for (int i = 0; i < n_peers_; ++i) {
        auto bs = std::make_shared<BinStream>();
        *bs << CommEvent::Scatter << vec[i];
        SendBinstreamTo(i, bs);
      }
    }
    auto my_msg = RecvBinstreamFromRoot();
    CommEvent e;
    T ret;
    *my_msg >> e >> ret;
    CHECK(e == CommEvent::Scatter);
    return ret;
  }
  template <typename T>
  void Broadcast(T &val) {
    if (IsRoot()) {
      for (int i = 0; i < n_peers_; ++i) {
        auto bs = std::make_shared<BinStream>();
        *bs << CommEvent::Broadcast << val;
        SendBinstreamTo(i, bs);
      }
    }
    auto my_msg = RecvBinstreamFromRoot();
    CommEvent e;
    *my_msg >> e >> val;
    CHECK(e == CommEvent::Broadcast);
  }
  template <typename T>
  T RingExchange(int needed_peer_id, const T &val) {
    auto bs = std::make_shared<BinStream>();
    *bs << CommEvent::RingExchange << rank_ << needed_peer_id << val;
    SendBinstreamTo(-1, bs);
    if (IsRoot()) {
      std::vector<T> gathered_val(n_peers_);
      std::vector<int> send_ids(n_peers_);
      for (int i = 0; i < n_peers_; ++i) {
        auto bs = RootRecvBinStream();
        CommEvent e;
        int peer_rank, needed_peer;
        T v;
        *bs >> e >> peer_rank >> needed_peer >> v;
        CHECK_EQ(e, CommEvent::RingExchange);
        send_ids[peer_rank] = needed_peer;
        gathered_val[peer_rank] = v;
      }
      for (int i = 0; i < n_peers_; ++i) {
        auto bs = std::make_shared<BinStream>();
        *bs << CommEvent::RingExchange << send_ids[i]
            << gathered_val[send_ids[i]];
        SendBinstreamTo(i, bs);
      }
    }
    auto my_msg = RecvBinstreamFromRoot();
    CommEvent e;
    int peer_id;
    T v;
    *my_msg >> e >> peer_id >> v;
    CHECK_EQ(peer_id, needed_peer_id);
    return v;
  }

 private:
  bool is_root_;
  int rank_;
  int n_peers_;
  zmq::context_t *zmq_ctx_;
  std::vector<std::unique_ptr<zmq::socket_t>> to_peers_;  // For root
  std::unique_ptr<zmq::socket_t> root_receiver_;          // For root

  std::unique_ptr<zmq::socket_t> send_to_root_;    // For all
  std::unique_ptr<zmq::socket_t> recv_from_root_;  // For all
  std::string recv_addr_;                          // For all
  std::vector<ProcInfo> peer_infos_;
};

}  // namespace gccl
