#include "comm/comm_info.h"

#include "gtest/gtest.h"

#include "base/bin_stream.h"
#include "comm/pattern/ring_comm_pattern.h"
#include "test_utils.h"

namespace gccl {
namespace {

class TestCommInfo : public testing::Test {
 public:
  TestCommInfo() {
    send_off = {0, 0, 1, 3};
    send_ids = {1, 2, 5};
    recv_off = {0, 2, 4, 4};
    recv_ids = {5, 2, 5, 3};
    max_comm_size = {2, 2, 2};

    pattern_info.type = Ring;
    auto *ring_info = pattern_info.GetRingCommPatternInfo();
    ring_info->n_stages = 3;
    ring_info->send_off = send_off.data();
    ring_info->send_ids = send_ids.data();
    ring_info->recv_off = recv_off.data();
    ring_info->recv_ids = recv_ids.data();
    ring_info->max_comm_size = max_comm_size.data();
  }

 protected:
  CommPatternInfo pattern_info;
  std::vector<int> send_off;
  std::vector<int> send_ids;
  std::vector<int> recv_off;
  std::vector<int> recv_ids;
  std::vector<int> max_comm_size;
};

TEST_F(TestCommInfo, SerAndDeser) {
  CommInfo info, de_info;
  info.allgather_scheme.n_blocks = 2;
  info.allgather_scheme.comm_pattern_infos[0] = pattern_info;
  info.allgather_scheme.comm_pattern_infos[1] = pattern_info;
  BinStream stream;
  info.serialize(stream);
  de_info.deserialize(stream);
  EXPECT_EQ(de_info.allgather_scheme.n_blocks, 2);
  auto *info0 =
      de_info.allgather_scheme.comm_pattern_infos[0].GetRingCommPatternInfo();
  auto *info1 =
      de_info.allgather_scheme.comm_pattern_infos[1].GetRingCommPatternInfo();

  EXPECT_EQ(info0->n_stages, 3);
  EXPECT_EQ(info1->n_stages, 3);
  EXPECT_VEC_EQ(info0->send_ids, send_ids);
  EXPECT_VEC_EQ(info0->send_off, send_off);
  EXPECT_VEC_EQ(info0->recv_ids, recv_ids);
  EXPECT_VEC_EQ(info0->recv_off, recv_off);
  EXPECT_VEC_EQ(info0->max_comm_size, max_comm_size);

  EXPECT_VEC_EQ(info1->send_ids, send_ids);
  EXPECT_VEC_EQ(info1->send_off, send_off);
  EXPECT_VEC_EQ(info1->recv_ids, recv_ids);
  EXPECT_VEC_EQ(info1->recv_off, recv_off);
  EXPECT_VEC_EQ(info1->max_comm_size, max_comm_size);
}

}  // namespace
}  // namespace gccl
