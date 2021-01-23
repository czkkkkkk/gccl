#include "gccl.h"

#include <memory>

#include "communicator.h"
#include "comm/comm_scheme.h"
#include "conn/net.h"
#include "conn/nvmlwrap.h"
#include "global_state.h"
#include "param.h"
#include "utils.h"

namespace gccl {

GlobalState gccl_global;

extern "C" __attribute__((visibility("default"))) gcclNet_t *gcclNet = NULL;
void InitNet() { gcclNet = gcclIbSupport() ? &gcclNetIb : nullptr; }

void Initialize() {
  if (gccl_global.initialized) return;
  InitNet();
  gccl_global.initialized = true;
  gccl_global.zmq_ctx = std::make_unique<zmq::context_t>();
  gccl_global.coordinator =
      std::make_unique<Coordinator>(gccl_global.zmq_ctx.get());
}

gcclUniqueId GetUniqueId() {
  int port = GetAvailablePort();
  std::string bind_addr = std::string("tcp://*:") + std::to_string(port);
  std::string conn_addr =
      std::string("tcp://") + GetHostName() + ":" + std::to_string(port);
  Initialize();
  gccl_global.coordinator->RootInit(StringToUniqueId(bind_addr));
  return StringToUniqueId(conn_addr);
}

gcclUniqueId GetUniqueId(const char *master, int port, bool is_root) {
  std::string bind_addr = std::string("tcp://*:") + std::to_string(port);
  std::string conn_addr =
      std::string("tcp://") + std::string(master) + ":" + std::to_string(port);
  if (is_root) {
    Initialize();
    gccl_global.coordinator->RootInit(StringToUniqueId(bind_addr));
  }
  return StringToUniqueId(conn_addr);
}

bool SetCpuAffinity(int cudaDev, nvmlDevice_t *nvmlDevice) {
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  if (cudaDeviceGetPCIBusId(busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
                            cudaDev) != cudaSuccess)
    return false;
  if (wrapNvmlDeviceGetHandleByPciBusId(busId, nvmlDevice) != gcclSuccess)
    return false;
  if (wrapNvmlDeviceSetCpuAffinity(*nvmlDevice) != gcclSuccess) {
    LOG(ERROR) << "Failed to set CPU affinity";
    return false;
  }
  return true;
}

Communicator *CreateCommunicator(Coordinator *coordinator, Config *config,
                                 int nranks, int rank) {
  gccl_global.comms.emplace_back(
      std::make_unique<Communicator>(coordinator, config, nranks, rank));
  return gccl_global.comms.back().get();
}

void CommInitRank(gcclComm_t *comm, int nranks, gcclUniqueId comm_id,
                  int rank) {
  Initialize();

  auto config_file = GetEnvParam("CONFIG", std::string(""));
  if (config_file == "") {
    gccl_global.config = std::make_unique<Config>(DefaultConfig(nranks));
  } else {
    gccl_global.config = std::make_unique<Config>(LoadConfig(config_file));
  }
  PrintConfig(*gccl_global.config);
  // int dev_id = rank;
  // if (gccl_global.config->rank_to_dev_id.count(rank) > 0) {
  //   dev_id = gccl_global.config->rank_to_dev_id[rank];
  // }
  gccl_global.coordinator->SetRankAndNPeers(rank, nranks);
  gccl_global.coordinator->BuildPeerInfo(comm_id);
  *comm = CreateCommunicator(gccl_global.coordinator.get(),
                             gccl_global.config.get(), nranks, rank);
  gccl_global.coordinator->Barrier();

#ifdef HAVE_CUDA
  cpu_set_t affinitySave;
  sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);

  CHECK(wrapNvmlSymbols());
  CHECK(wrapNvmlInit());
  // Make sure all host memory allocation are close to the GPU
  int cudaDev = gccl_global.coordinator->GetDevId();
  cudaSetDevice(cudaDev);
  nvmlDevice_t nvmlDevice;
  
  SetCpuAffinity(cudaDev, &nvmlDevice);

  // sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  // wrapNvmlShutdown();
#endif
}

}  // namespace gccl
