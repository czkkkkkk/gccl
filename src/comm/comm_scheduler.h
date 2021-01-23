#pragma once

#include <vector>

#include "comm/comm_info.h"
#include "config.h"
#include "conn/connection.h"
#include "coordinator.h"
#include "graph.h"

namespace gccl {

class CommScheduler {
 public:
  const std::vector<std::map<int, int>> &GetLocalMappings() const {
    return local_mappings_;
  }
  int GetLocalNNodes() const { return my_local_graph_info_.n_local_nodes; }
  const std::vector<std::shared_ptr<CommPattern>> &GetCommPatterns() const {
    return comm_patterns_;
  }

  void BuildPartitionInfo(Coordinator *coor, Config *config, Graph &g,
                          const std::string &graph_dir, CommInfo **info,
                          int *sgn, int **sg_xadj, int **sg_adjncy);

  void BuildLocalMappings(Graph &g, int nparts, const std::vector<int> &parts);

  TransferRequest BuildTransferRequest(Graph &g, int nparts,
                                       const std::vector<int> &parts);

  std::vector<TransferRequest> AllocateRequestToBlock(
      const TransferRequest &all_req, int n_parts, int n_blocks);
  CommInfo *ScatterCommInfo(Coordinator *coordinator,
                            const std::vector<CommInfo> &infos);

  std::vector<Graph> BuildSubgraphs(
      const Graph &g, const std::vector<std::map<int, int>> &local_mappings,
      const std::vector<int> &parts, int nparts);

  void DispatchData(Coordinator *coor, char *data, int feat_size, int data_size,
                    int local_n_nodes, char *local_data, int no_remote);

  void LoadCachedPartition(Coordinator *coor, const std::string &dir, int *sgn,
                           int **sg_xadj, int **sg_adjncy);

 private:
  void ReadCachedState(const std::string &part_dir, int rank, bool is_root);
  void WriteCachedState(Coordinator* coor, const std::string &part_dir, int rank, bool is_root);
  void ScatterLocalGraphInfos(Coordinator *coor);
  void PartitionGraph(Coordinator *coor, Graph &g, const std::string &dir,
                      int n_parts, int *sgn, int **sg_xadj, int **sg_adjncy);

  std::vector<std::shared_ptr<CommPattern>> comm_patterns_;

  std::vector<std::map<int, int>> local_mappings_;  // All
  std::vector<int> parts_;                          // All
  TransferRequest requests_;
  std::vector<LocalGraphInfo> all_local_graph_infos_;  // All
  LocalGraphInfo my_local_graph_info_;                 // Mine
  Graph my_graph_;                                     // Mine
};

}  // namespace gccl
