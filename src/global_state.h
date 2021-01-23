#pragma once

#include <memory>

#include "zmq.hpp"

#include "config.h"
#include "coordinator.h"

namespace gccl {

struct GlobalState {
  std::unique_ptr<zmq::context_t> zmq_ctx = nullptr;

  std::unique_ptr<Coordinator> coordinator = nullptr;

  bool initialized = false;
  std::vector<std::unique_ptr<Communicator>> comms;

  std::unique_ptr<Config> config;
};

};  // namespace gccl