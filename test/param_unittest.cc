#include "param.h"

#include <cstdlib>
#include "gtest/gtest.h"

namespace gccl {
namespace {

class TestParam : public testing::Test {};

TEST_F(TestParam, function) {
  setenv("GCCL_TEST_ENV", "1", 1);
  setenv("GCCL_TEST_ENV2", "123", 1);
  int v = GetEnvParam<int>("TEST_ENV", -1);
  uint64_t v2 = GetEnvParam<uint64_t>("TEST_ENV2", -3);
  int v3 = GetEnvParam<uint64_t>("TEST_ENV3", -3);
  EXPECT_EQ(v, 1);
  EXPECT_EQ(v2, 123);
  EXPECT_EQ(v3, -3);
}

}  // namespace
}  // namespace gccl