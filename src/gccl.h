#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>

// #include "comm/comm_info.h"
// #include "communicator.h"

namespace gccl {

class Communicator;
struct CommInfo;

// GCCL communication, which store the communication environment and graph partition information
typedef Communicator *gcclComm_t;
// GCCL communication information
typedef CommInfo *gcclCommInfo_t;

typedef enum {
  gcclSuccess = 0,
  gcclUnhandledCudaError = 1,
  gcclSystemError = 2,
  gcclInternalError = 3,
  gcclInvalidArgument = 4,
  gcclInvalidUsage = 5,
  gcclNumResults = 6
} gcclResult_t;

#define MAX_COMM_ID_LEN 128

// GCCL communication id
struct gcclUniqueId {
  char data[MAX_COMM_ID_LEN];
};

typedef enum {
  gcclFloat = 0,
  gcclInt = 1,
} gcclDataType_t;

// Construct a unique id for communication as in NCCL.
// This API is used in MPI mode. The root process build a 
// unique id and then broadcast to other processes.
gcclUniqueId GetUniqueId();
// Construct a unique id for communication as in NCCL.
// This API is used when different processes are communicate
// using TCP.
gcclUniqueId GetUniqueId(const char *master, int port,
                         bool is_root); 

// Initialize the log directory
void InitLogs(const char *file);

int GetDeviceId(gcclComm_t comm);

// Initialize a communication environment and store the communicator in `comm`.
// Each process specifies the size of the communication group `nranks`, the same communication
// id `comm_id` and their own rank.
void CommInitRank(gcclComm_t *comm, int nranks, gcclUniqueId comm_id, int rank);

// Partition a graph under the communicator `comm`.
// `(n, xadj, adjncy)` is the csr graph format which is provided  by the root process
// After partition, each process will have the communication information `info` and
// their own subgraphs `(sgn, sg_xadj, sg_adjncy)`.
void PartitionGraph(gcclComm_t comm, int n, int *xadj, int *adjncy,
                    gcclCommInfo_t *info, int *sgn, int **sg_xadj,
                    int **sg_adjncy);

// Partition a graph under the communicator 'comm'.
// It will partition the graph `cache_dir/graph.txt` if the `cache_dir/part-k` does not exist, where k is the number of processes.
// Otherwise it will read the partition cache `cache_dir/part-k`.
// After partition ,each process will have their communication scheme for the graph in `info`.
void PartitionGraph(gcclComm_t comm, const char *cached_dir,
                    gcclCommInfo_t *info, int *sgn, int **sg_xadj,
                    int **sg_adjncy);

int GetLocalNNodes(gcclComm_t comm);

// TODO
void FreeCommInfo(gcclCommInfo_t info);

// Dispatch float feature
// `data` is the overall feature provide by the root process
// After dispatch, each process will have their own feature in `local_data` according to the partition scheme
void DispatchFloat(gcclComm_t comm, float *data, int feat_size,
                   int local_n_nodes, float *local_data, int no_remote);

void DispatchInt(gcclComm_t comm, int *data, int feat_size, int local_n_nodes,
                 int *local_data, int no_remote);

// Performance one allgather operation
// `comm` and `info` are communicator and graph communication scheme for each process
// `input` are the local vertex data on each process on GPU. It should also reserve enough space for the remote vertex data
// After `graphAllgather` each process will have the feature 
void GraphAllgather(gcclComm_t comm, gcclCommInfo_t info, void *input,
                    gcclDataType_t type, int feat_size, cudaStream_t stream);

// Performance the backward of the allgather operation
// `comm` and `info` are communicator and graph communication scheme for each process
// `input` is the vertex gradients including remote vertices gradients
// After `GraphAllgatherBackward`, the local vertices of each process will gather their gradients from remote neighbors.
void GraphAllgatherBackward(gcclComm_t comm, gcclCommInfo_t info, void *input,
                            gcclDataType_t type, int feat_size,
                            cudaStream_t stream);

std::pair<std::vector<int>, std::vector<int>> RandGraph(int n, int m);
std::pair<std::vector<int>, std::vector<int>> ReadGraph(
    const std::string &file);

void SetConfig(gcclComm_t comm, const std::string &config_json);

}  // namespace gccl
