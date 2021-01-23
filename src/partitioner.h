#pragma once

#include <vector>

#include "graph.h"

namespace gccl {

void PartitionGraphMetis(int n, int *xadj, int *adjncy, int nparts, int *objval,
                         int *parts);

// Return parts
std::vector<int> PartitionGraphInternal(Graph &graph, int nparts);

}  // namespace gccl