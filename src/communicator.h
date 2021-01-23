#pragma once

#include "config.h"
#include "coordinator.h"
#include "comm/comm_scheduler.h"

namespace gccl {

// Communicator holds all the communication information in CPU
class Communicator {
 public:
  Communicator(Coordinator *coordinator, Config *config, int nranks, int rank);

  Coordinator *GetCoordinator() const { return coordinator_; }
  Config *GetConfig() const { return config_; }

  CommScheduler *GetCommScheduler() const;

  int GetRank() const { return rank_; }
  int GetNRanks() const { return nranks_; }

 private:
  Coordinator *coordinator_;
  Config *config_;
  int nranks_;
  int rank_;
  std::unique_ptr<CommScheduler> comm_scheduler_;
};

}  // namespace gccl
