#include "comm/comm_info.h"

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include "comm/pattern/ring_comm_pattern.h"
#include "gpu/common.h"
#include "test_utils.h"

namespace gccl {
namespace {

class TestGPUCommInfo : public testing::Test {
 public:
  TestGPUCommInfo() {
    send_off = {0, 0, 1, 3};
    send_ids = {1, 2, 5};
    recv_off = {0, 2, 4, 4};
    recv_ids = {5, 2, 5, 3};
    max_comm_size = {2, 2};
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

TEST_F(TestGPUCommInfo, CopyPatternInfoToGPU) {
  CommPatternInfo gpu_info = pattern_info;

  cudaSetDevice(0);
  gpu_info.CopyGraphInfoToDev();
  auto *ring_info = gpu_info.GetRingCommPatternInfo();
  EXPECT_GPU_CPU_VEC_EQ(ring_info->send_off, send_off);
  EXPECT_GPU_CPU_VEC_EQ(ring_info->send_ids, send_ids);
  EXPECT_GPU_CPU_VEC_EQ(ring_info->recv_off, recv_off);
  EXPECT_GPU_CPU_VEC_EQ(ring_info->recv_ids, recv_ids);
}

TEST_F(TestGPUCommInfo, CopyGraphInfoToDev) {
  CommInfo info;
  info.allgather_scheme.n_blocks = 2;
  info.allgather_scheme.comm_pattern_infos[0] = pattern_info;
  info.allgather_scheme.comm_pattern_infos[1] = pattern_info;
  cudaSetDevice(0);
  info.CopyGraphInfoToDev();
  auto *pattern_info0 =
      info.allgather_scheme.comm_pattern_infos[0].GetRingCommPatternInfo();
  auto *pattern_info1 =
      info.allgather_scheme.comm_pattern_infos[1].GetRingCommPatternInfo();

  EXPECT_GPU_CPU_VEC_EQ(pattern_info0->send_off, send_off);
  EXPECT_GPU_CPU_VEC_EQ(pattern_info0->send_ids, send_ids);
  EXPECT_GPU_CPU_VEC_EQ(pattern_info0->recv_off, recv_off);
  EXPECT_GPU_CPU_VEC_EQ(pattern_info0->recv_ids, recv_ids);
  EXPECT_GPU_CPU_VEC_EQ(pattern_info0->max_comm_size, max_comm_size);

  EXPECT_GPU_CPU_VEC_EQ(pattern_info1->send_off, send_off);
  EXPECT_GPU_CPU_VEC_EQ(pattern_info1->send_ids, send_ids);
  EXPECT_GPU_CPU_VEC_EQ(pattern_info1->recv_off, recv_off);
  EXPECT_GPU_CPU_VEC_EQ(pattern_info1->recv_ids, recv_ids);
  EXPECT_GPU_CPU_VEC_EQ(pattern_info1->max_comm_size, max_comm_size);
}

}  // namespace
}  // namespace gccl
