#include "partitioner.h"

#include "glog/logging.h"
#include "gtest/gtest.h"

namespace gccl {
namespace {

class TestPartitioner : public testing::Test {};

TEST_F(TestPartitioner, functional) {
  int n = 5;
  int m = 7;
  int nparts = 3;
  int xadj[] = {0, 4, 4, 4, 6, 7};
  int adjncy[] = {1, 2, 3, 4, 2, 4, 1};
  int objval = 0;
  int *part = new int[n];
  PartitionGraphMetis(n, xadj, adjncy, nparts, &objval, part);
  LOG(INFO) << "Objval " << objval;
  for (int i = 0; i < n; ++i) {
    LOG(INFO) << "Part " << i << " is " << part[i];
    EXPECT_GE(part[i], 0);
    EXPECT_LT(part[i], nparts);
  }
}

}  // namespace
}  // namespace gccl