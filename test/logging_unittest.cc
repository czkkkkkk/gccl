#include <cstdio>

#include "glog/logging.h"
#include "gtest/gtest.h"

class TestLogging : public testing::Test {};

TEST_F(TestLogging, functional) { LOG(INFO) << "Google Log Test"; }