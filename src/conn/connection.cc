#include "connection.h"

#include "glog/logging.h"

#include "comm/pattern/comm_pattern.h"
#include "conn/nvlink.h"
#include "conn/nvmlwrap.h"
#include "conn/p2p_connection.h"
#include "conn/net_connection.h"
#include "conn/shm_connection.h"
#include "conn/topo.h"
#include "gpu/common.h"
#include "param.h"

namespace gccl {

#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE 32

void GetBusId(char bus_id[], int dev_id) {
  CUDACHECK(cudaDeviceGetPCIBusId(bus_id, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
                                  dev_id));
  nvmlDevice_t nvmlDevice;
  CHECK(wrapNvmlDeviceGetHandleByPciBusId(bus_id, &nvmlDevice));
  nvmlPciInfo_t pciInfo;
  CHECK(wrapNvmlDeviceGetHandleByPciInfo(nvmlDevice, &pciInfo));
  strncpy(bus_id, pciInfo.busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE);
}

bool CanP2p(const ProcInfo& my_info, const ProcInfo& peer_info) {
  /* Determine if we can communicate with the peer through p2p */
  // Do not use P2P across root complexes by default (provided CUDA permits
  // it)
  int p2pLevel = PATH_SOC;
  std::string tr_level = GetEnvParam("TRANSPORT_LEVEL", std::string("p2p"));
  if (tr_level != "p2p") return false;

  // Rule out different nodes
  if (my_info.hostname != peer_info.hostname) return false;

  // Do not detect topology if we're on the same GPU. Note this is not really
  // supported.
  if (my_info.dev_id == peer_info.dev_id) {
    return true;
  }

  // See if CUDA can do P2P
  int p2p;
  if (cudaDeviceCanAccessPeer(&p2p, my_info.dev_id, peer_info.dev_id) !=
      cudaSuccess) {
    LOG(FATAL) << "peer query failed between dev" << my_info.dev_id
               << " and dev " << peer_info.dev_id;
    return false;
  }
  if (p2p == 0) return false;

  // Check for NVLink/NVswitch
  char my_bus_id[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE],
      peer_bus_id[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  GetBusId(my_bus_id, my_info.dev_id);
  GetBusId(peer_bus_id, peer_info.dev_id);
  int nvlinkp2p = getNvlinkGpu(my_bus_id, peer_bus_id);
  if (nvlinkp2p > 0) {
    return true;
  }

  // Finally compute the PCI distance and compare with the p2pLevel.
  char* myPath;
  char* peerPath;
  gcclResult_t suc1 = getCudaPath(my_info.dev_id, &myPath);
  gcclResult_t suc2 = getCudaPath(peer_info.dev_id, &peerPath);
  int ret = 0;
  if (suc1 == gcclSuccess && suc2 == gcclSuccess) {
    int distance = pciDistance(myPath, peerPath);
    if (distance < p2pLevel) {
      ret = 1 + PATH_SOC - distance;
    }
  } else {
    LOG(INFO) << "GetCudaPath unsuccessful";
  }
  if (suc1) free(myPath);
  if (suc2) free(peerPath);
  return ret;
}

ConnType GetConnType(const ProcInfo& my_info, const ProcInfo& peer_info) {
  if(my_info.hostname != peer_info.hostname) {
    return ConnType::IB;
  }

  if (CanP2p(my_info, peer_info)) {
    LOG(INFO) << "Device " << my_info.dev_id << " can p2p access "
              << peer_info.dev_id;
    return ConnType::P2P;
  } else {
    LOG(INFO) << "Device " << my_info.dev_id << " can not p2p access "
              << peer_info.dev_id;
    return ConnType::SHM;
  }
}

std::shared_ptr<Connection> ConnectionFactory::GetConnection(
    ConnType type, const ProcInfo& my_info, const ProcInfo& peer_info,
    int bid) {
  switch (type) {
    case ConnType::P2P:
      return std::make_shared<P2pConnection>(my_info, peer_info, bid);
    case ConnType::SHM:
      return std::make_shared<ShmConnection>(my_info, peer_info, bid);
    case ConnType::IB:
      return std::make_shared<NetConnection>(my_info, peer_info, bid);
    default:
      CHECK(false);
  }
  return nullptr;
}

}  // namespace gccl
