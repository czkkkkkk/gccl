#include "glog/logging.h"
#include "gtest/gtest.h"

#ifdef MPI_TEST
#include "mpi.h"
#endif

GTEST_API_ int main(int argc, char **argv) {
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);
  testing::InitGoogleTest(&argc, argv);
#ifdef MPI_TEST
  LOG(INFO) << "MPI test is enabled";
  MPI_Init(&argc, &argv);
#else
  LOG(INFO) << "MPI test is not enabled";
#endif

  int result = RUN_ALL_TESTS();

#ifdef MPI_TEST
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  return result;
}