/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include "core.h"

#include "glog/logging.h"
#include "nvml.h"

#define NVMLCHECK(cmd)                                            \
  do {                                                            \
    nvmlReturn_t e = cmd;                                         \
    if (e != NVML_SUCCESS) {                                      \
      LOG(INFO) << "NVML failure '" << nvmlErrorString(e) << "'"; \
      return false;                                               \
    }                                                             \
  } while (false)

static bool wrapNvmlSymbols(void) { return true; }
static bool wrapNvmlInit(void) {
  NVMLCHECK(nvmlInit());
  return true;
}
static bool wrapNvmlShutdown(void) {
  NVMLCHECK(nvmlShutdown());
  return true;
}
static bool wrapNvmlDeviceGetHandleByPciBusId(const char *pciBusId,
                                              nvmlDevice_t *device) {
  NVMLCHECK(nvmlDeviceGetHandleByPciBusId(pciBusId, device));
  return true;
}
static bool wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned *index) {
  NVMLCHECK(nvmlDeviceGetIndex(device, index));
  return true;
}
static bool wrapNvmlDeviceSetCpuAffinity(nvmlDevice_t device) {
  NVMLCHECK(nvmlDeviceSetCpuAffinity(device));
  return true;
}
static bool wrapNvmlDeviceClearCpu(nvmlDevice_t device) {
  NVMLCHECK(nvmlDeviceClearCpuAffinity(device));
  return true;
}
static bool wrapNvmlDeviceGetHandleByIndex(unsigned int index,
                                           nvmlDevice_t *device) {
  NVMLCHECK(nvmlDeviceGetHandleByIndex(index, device));
  return true;
}
static bool wrapNvmlDeviceGetHandleByPciInfo(nvmlDevice_t device,
                                             nvmlPciInfo_t *pci) {
  NVMLCHECK(nvmlDeviceGetPciInfo(device, pci));
  return true;
}
static bool wrapNvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link,
                                         nvmlEnableState_t *isActive) {
  NVMLCHECK(nvmlDeviceGetNvLinkState(device, link, isActive));
  return true;
}
static bool wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device,
                                                 unsigned int link,
                                                 nvmlPciInfo_t *pci) {
  NVMLCHECK(nvmlDeviceGetNvLinkRemotePciInfo(device, link, pci));
  return true;
}
static bool wrapNvmlDeviceGetNvLinkCapability(nvmlDevice_t device,
                                              unsigned int link,
                                              nvmlNvLinkCapability_t capability,
                                              unsigned int *capResult) {
  NVMLCHECK(nvmlDeviceGetNvLinkCapability(device, link, capability, capResult));
  return true;
}