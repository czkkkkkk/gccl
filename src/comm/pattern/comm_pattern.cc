#include "comm_pattern.h"

#include <stddef.h>

#include "glog/logging.h"
#include "nlohmann/json.hpp"

#include "base/bin_stream.h"
#include "comm/pattern/all_to_all_comm_pattern.h"
#include "comm/pattern/greedy_comm_pattern.h"
#include "comm/pattern/ring_comm_pattern.h"
#include "conn/connection.h"
#include "core.h"
#include "gpu/common.h"
#include "param.h"
#include "utils.h"

namespace gccl {

void PrintPtrAlign(void *ptr, const std::string &name, int align) {
  bool is_align = (unsigned long)ptr % align == 0;
  LOG(INFO) << name << " is " << (is_align ? "" : "not ") << "aligned with "
            << align << " bytes";
}

BinStream &CommPatternInfo::serialize(BinStream &stream) const {
  stream << type;
  switch (type) {
    case Ring: {
      auto *ring_info = (RingCommPatternInfo *)data;
      ring_info->serialize(stream);
      break;
    }
    case AllToAll: {
      auto *aa_info = (AllToAllCommPatternInfo *)data;
      aa_info->serialize(stream);
      break;
    }
    case Greedy: {
      auto *greedy_info = (GreedyCommPatternInfo *)data;
      greedy_info->serialize(stream);
      break;
    }
    default:
      CHECK(false);
  }
  return stream;
}

BinStream &CommPatternInfo::deserialize(BinStream &stream) {
  stream >> type;
  switch (type) {
    case Ring: {
      auto *ring_info = (RingCommPatternInfo *)data;
      ring_info->deserialize(stream);
      break;
    }
    case AllToAll: {
      auto *aa_info = (AllToAllCommPatternInfo *)data;
      aa_info->deserialize(stream);
      break;
    }
    case Greedy: {
      auto *greedy_info = (GreedyCommPatternInfo *)data;
      greedy_info->deserialize(stream);
      break;
    }
    default:
      CHECK(false);
  }
  return stream;
}

void CommPatternInfo::CopyGraphInfoToDev() {
  switch (type) {
    case Ring: {
      auto *ring_comm_pattern_info = GetRingCommPatternInfo();
      ring_comm_pattern_info->CopyGraphInfoToDev();
      break;
    }
    case AllToAll: {
      auto *aa_info = GetAllToAllCommPatternInfo();
      aa_info->CopyGraphInfoToDev();
      break;
    }
    case Greedy: {
      auto *greedy_info = GetGreedyCommPatternInfo();
      greedy_info->CopyGraphInfoToDev();
      break;
    }
    default:
      CHECK(false);
  }
}

void CommPatternInfo::Print() const {
#ifdef GCCL_DEBUG
  switch (type) {
    case Ring: {
      auto *ring_comm_pattern_info = (RingCommPatternInfo *)data;
      ring_comm_pattern_info->Print();
      break;
    }
    case AllToAll: {
      // auto *aa_info = GetAllToAllCommPatternInfo();
      // aa_info->Print();
      LOG(INFO) << "All to all comm do not have print function";
      break;
    }
    case Greedy: {
      auto *greedy_info = GetGreedyCommPatternInfo();
      greedy_info->Print();
      break;
    }
    default:
      CHECK(false);
  }
#endif
}
int CommPatternInfo::GetMemBytes() const {
  switch(type) {
    case Greedy: {
      auto *greedy_info = GetGreedyCommPatternInfo();
      return greedy_info->GetMemBytes();
    }
    default:
      return 0;
  }
}

RingCommPatternInfo *CommPatternInfo::GetRingCommPatternInfo() const {
  CHECK(type == Ring);
  return (RingCommPatternInfo *)data;
}

AllToAllCommPatternInfo *CommPatternInfo::GetAllToAllCommPatternInfo() const {
  CHECK(type == AllToAll);
  return (AllToAllCommPatternInfo *)data;
}

GreedyCommPatternInfo *CommPatternInfo::GetGreedyCommPatternInfo() const {
  CHECK(type == Greedy);
  return (GreedyCommPatternInfo *)data;
}

std::shared_ptr<CommPattern> CommPatternFactory::GetCommPattern(
    CommPatternType type, const std::vector<int> &topo,
    const std::vector<ConnType> &conn_type) {
  switch (type) {
    case Ring:
      return std::make_shared<RingCommPattern>(topo, conn_type);
    case AllToAll:
      return std::make_shared<AllToAllCommPattern>(topo, conn_type);
    case Greedy:
      return std::make_shared<GreedyCommPattern>(topo, conn_type);
    default:
      CHECK(false);
  }
  return nullptr;
}

}  // namespace gccl
