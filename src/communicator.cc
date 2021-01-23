#include "communicator.h"

#include "comm/comm_scheduler.h"

namespace gccl {

Communicator::Communicator(Coordinator* coordinator, Config* config, int nranks,
                           int rank)
    : coordinator_(coordinator), config_(config), nranks_(nranks), rank_(rank) {
  comm_scheduler_ = std::make_unique<CommScheduler>();
}

CommScheduler *Communicator::GetCommScheduler() const { return comm_scheduler_.get(); }

}  // namespace gccl
