#include "gccl.h"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "communicator.h"
#include "config.h"
#include "graph.h"
#include "param.h"
#include "utils.h"

namespace gccl {

void PartitionGraph(gcclComm_t comm, int n, int *xadj, int *adjncy,
                    gcclCommInfo_t *info, int *sgn, int **sg_xadj,
                    int **sg_adjncy) {
  // To internal graph
  // Pass to scheduler
  Graph g;
  if (comm->GetCoordinator()->IsRoot()) {
    g = Graph(n, xadj, adjncy);
  }
  auto *comm_sch = comm->GetCommScheduler();
  auto *coor = comm->GetCoordinator();
  auto *config = comm->GetConfig();
  comm_sch->BuildPartitionInfo(coor, config, g, "", info, sgn, sg_xadj,
                               sg_adjncy);
}

void PartitionGraph(gcclComm_t comm, const char *cached_dir,
                    gcclCommInfo_t *info, int *sgn, int **sg_xadj,
                    int **sg_adjncy) {
  auto *comm_sch = comm->GetCommScheduler();
  auto *coor = comm->GetCoordinator();
  auto *config = comm->GetConfig();
  Graph g;
  comm_sch->BuildPartitionInfo(coor, config, g, cached_dir, info, sgn, sg_xadj,
                               sg_adjncy);
}

void InitLogs(const char *name) {
  google::InitGoogleLogging(name);
  LOG(INFO) << "Init logs name is " << name;
}

int GetDeviceId(gcclComm_t comm) { return comm->GetCoordinator()->GetDevId(); }

int GetLocalNNodes(gcclComm_t comm) {
  auto *comm_sch = comm->GetCommScheduler();
  return comm_sch->GetLocalNNodes();
}
void FreeCommInfo(gcclCommInfo_t info) {
  // TODO
}

void DispatchFloat(gcclComm_t comm, float *data, int feat_size,
                   int local_n_nodes, float *local_data, int no_remote) {
  auto *comm_sch = comm->GetCommScheduler();
  auto *coor = comm->GetCoordinator();
  comm_sch->DispatchData(coor, (char *)data, feat_size, sizeof(float),
                         local_n_nodes, (char *)local_data, no_remote);
}

void DispatchInt(gcclComm_t comm, int *data, int feat_size, int local_n_nodes,
                 int *local_data, int no_remote) {
  auto *comm_sch = comm->GetCommScheduler();
  auto *coor = comm->GetCoordinator();
  comm_sch->DispatchData(coor, (char *)data, feat_size, sizeof(int),
                         local_n_nodes, (char *)local_data, no_remote);
}

void SetConfig(gcclComm_t comm, const std::string &config_json) {
  SetConfigInternal(comm->GetConfig(), config_json);
}

}  // namespace gccl
