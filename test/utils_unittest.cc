#include "utils.h"

#include "gtest/gtest.h"

#include "test_utils.h"

namespace gccl {
namespace {

class TestUtils : public testing::Test {};

TEST_F(TestUtils, UniqueVec) {
  std::vector<int> vec({1, 5, 2, 3, 2, 1});
  UniqueVec(vec);
  EXPECT_VEC_EQ(vec, {1, 2, 3, 5});
}

TEST_F(TestUtils, CopyVectorToRawPtr) {
  std::vector<int> vec({3, 2, 1});
  int *a = new int[3];
  CopyVectorToRawPtr(&a, vec);
  for (int i = 0; i < 3; ++i) EXPECT_EQ(a[i], vec[i]);
  delete a;
}

}  // namespace
}  // namespace gccl