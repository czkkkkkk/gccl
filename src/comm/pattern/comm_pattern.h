#pragma once

#include <map>
#include <vector>

#include "base/bin_stream.h"
#include "config.h"
#include "conn/connection.h"
#include "coordinator.h"

namespace gccl {

#define MAX_COMM_PATTERN_INFO_SIZE 512


struct TransferRequest {
  // rank, rank, transfer id
  std::vector<std::vector<std::vector<int>>> req_ids;
};

enum CommPatternType { Ring, AllToAll, Greedy };

struct RingCommPatternInfo;
struct AllToAllCommPatternInfo;
struct GreedyCommPatternInfo;

struct __align__(16) CommPatternInfo {
  char data[MAX_COMM_PATTERN_INFO_SIZE];
  union {
    CommPatternType type;
    struct {
      char pad[16 - sizeof(CommPatternType)];
    };
  };

  BinStream &serialize(BinStream & stream) const;
  BinStream &deserialize(BinStream & stream);
  void CopyGraphInfoToDev();
  void Print() const;
  RingCommPatternInfo *GetRingCommPatternInfo() const;
  AllToAllCommPatternInfo *GetAllToAllCommPatternInfo() const;
  GreedyCommPatternInfo *GetGreedyCommPatternInfo() const;
  int GetMemBytes() const;
};

class CommPattern {
 public:
  CommPattern(const std::vector<int> &dev_topo,
              const std::vector<ConnType> &conn_type)
      : dev_topo_(dev_topo), conn_type_(conn_type) {
    dev_topo_rmap_.resize(dev_topo_.size());
    for (int i = 0; i < dev_topo_.size(); ++i) {
      dev_topo_rmap_[dev_topo_[i]] = i;
    }
  }
  virtual ~CommPattern(){};

  virtual std::vector<CommPatternInfo> BuildCommPatternInfos(
      Config *config, const std::vector<std::map<int, int>> &local_mappings,
      const TransferRequest &req, int nparts) = 0;
  virtual void SetupConnection(CommPatternInfo *info, Coordinator *coor,
                               int bid, const std::vector<int> &conn_peers) = 0;

  virtual void StartProxy(Coordinator *coor, CommPatternInfo *info) = 0;
  virtual void SaveProxy(Coordinator *coor, CommPatternInfo *info,
                         int feat_size, int n_threads, bool forward) = 0;

  const std::vector<int> &GetDevTopo() const { return dev_topo_; }

 protected:
  std::vector<ConnType> conn_type_;
  std::vector<int> dev_topo_;
  std::vector<int> dev_topo_rmap_;
};

class CommPatternFactory {
 public:
  static std::shared_ptr<CommPattern> GetCommPattern(
      CommPatternType type, const std::vector<int> &topo,
      const std::vector<ConnType> &conn_type);
};

}  // namespace gccl
