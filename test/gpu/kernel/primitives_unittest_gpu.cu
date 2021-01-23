#include "gpu/kernel/primitives.h"

#include <algorithm>
#include <cstdlib>
#include <thread>
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "comm/pattern/ring_comm_pattern.h"
#include "gpu/common.h"
#include "test_cuda_utils.h"
#include "test_utils.h"
#include "utils.h"

namespace gccl {

namespace {

class TestPrimitives : public testing::Test {};

TEST_F(TestPrimitives, Copy128b) {
  int record_size = 32;
  int buff_size = 1024;
  int n_peers = 3;
  int n_int = record_size / sizeof(int);
  std::vector<CommPatternInfo> infos(3);
  std::vector<RingCommPatternInfo *> ring_infos;
  for (auto &info : infos) {
    ring_infos.push_back(info.GetRingCommPatternInfo());
  }
  std::vector<std::pair<int, int>> send_recv_size = {{3, 2}, {2, 3}, {2, 2}};

  cudaSetDevice(0);

  GCCLMallocAndCopy(&ring_infos[0]->recv_ids, {1, -2});
  GCCLMallocAndCopy(&ring_infos[1]->recv_ids, {0, -1, 1});
  GCCLMallocAndCopy(&ring_infos[2]->recv_ids, {2, -2});

  GCCLMallocAndCopy(&ring_infos[0]->send_ids, {0, 2, -1});
  GCCLMallocAndCopy(&ring_infos[1]->send_ids, {2, -2});
  GCCLMallocAndCopy(&ring_infos[2]->send_ids, {0, -1});

  std::vector<int *> inputs(n_peers);

  GCCLMallocAndCopy(&inputs[0], Repeat(std::vector<int>({5, -1, 9}), n_int));
  GCCLMallocAndCopy(&inputs[1], Repeat(std::vector<int>({-1, -1, 8}), n_int));
  GCCLMallocAndCopy(&inputs[2], Repeat(std::vector<int>({7, 1, -1}), n_int));

  GCCLMallocAndCopy((int **)&ring_infos[0]->dev_extra_mem,
                    Repeat(std::vector<int>({3, -1}), n_int));
  GCCLMallocAndCopy((int **)&ring_infos[1]->dev_extra_mem,
                    Repeat(std::vector<int>({-1, 4}), n_int));
  GCCLMallocAndCopy((int **)&ring_infos[2]->dev_extra_mem,
                    Repeat(std::vector<int>({0, -1}), n_int));

  int recv_dev_mem_size = offsetof(RecvDevMem, buff) + buff_size;
  for (int i = 0; i < n_peers; ++i) {
    GCCLCudaMalloc((char **)&ring_infos[i]->forward_conn.recv_dev_mem,
                   recv_dev_mem_size);
    GCCLCudaMalloc(&ring_infos[i]->forward_conn.send_dev_mem, 1);
  }
  for (int i = 0; i < n_peers; ++i) {
    auto &next_info = ring_infos[(i + 1) % n_peers];
    auto &prev_info = ring_infos[(i + n_peers - 1) % n_peers];
    auto &my_info = ring_infos[i];
    my_info->forward_conn.conn_info.my_stage_ready =
        &my_info->forward_conn.recv_dev_mem->stage_ready;
    my_info->forward_conn.conn_info.my_substage_ready =
        &my_info->forward_conn.recv_dev_mem->substage_ready;
    my_info->forward_conn.conn_info.my_substage_done =
        &my_info->forward_conn.send_dev_mem->substage_done;

    my_info->forward_conn.conn_info.next_recv_buff =
        &next_info->forward_conn.recv_dev_mem->buff;
    my_info->forward_conn.conn_info.next_substage_ready =
        &next_info->forward_conn.recv_dev_mem->substage_ready;
    my_info->forward_conn.conn_info.prev_substage_done =
        &prev_info->forward_conn.send_dev_mem->substage_done;
  }
  std::vector<std::vector<int>> expected_inputs = {
      Repeat(std::vector<int>({5, 7, 9}), n_int),
      Repeat(std::vector<int>({5, 3, 8}), n_int),
      Repeat(std::vector<int>({7, 1, 8}), n_int)};
  std::vector<std::vector<int>> expected_extra_buff = {
      Repeat(std::vector<int>({3, 0}), n_int),
      Repeat(std::vector<int>({9, 4}), n_int),
      Repeat(std::vector<int>({0, 4}), n_int)};

  std::vector<std::thread> ths;
  for (int i = 0; i < n_peers; ++i) {
    ths.emplace_back([&ring_infos, i, record_size, buff_size, &send_recv_size,
                      &inputs, &expected_inputs, &expected_extra_buff]() {
      CopyArgs args(-1, -1,
                    ring_infos[i]->forward_conn.conn_info.my_substage_ready,
                    ring_infos[i]->forward_conn.conn_info.my_substage_done,
                    ring_infos[i]->forward_conn.conn_info.next_substage_ready,
                    ring_infos[i]->forward_conn.conn_info.prev_substage_done);
      args.n_128b = record_size / PACK_SIZE;
      args.buff_n_128b = buff_size / PACK_SIZE;
      args.input = (Pack128 *)inputs[i];
      args.recv_buff =
          (Pack128 *)&ring_infos[i]->forward_conn.recv_dev_mem->buff;
      args.extra_buff = (Pack128 *)ring_infos[i]->dev_extra_mem;
      args.next_recv_buff =
          (Pack128 *)ring_infos[i]->forward_conn.conn_info.next_recv_buff;
      args.send_ids = ring_infos[i]->send_ids;
      args.send_size = send_recv_size[i].first;
      args.recv_ids = ring_infos[i]->recv_ids;
      args.recv_size = send_recv_size[i].second;
      args.max_comm_size = 3;
      args.extra_buff_size = 2;
      int n_threads = 32;
      void *kernel_args[] = {&args};
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      cudaLaunchKernel((void *)Copy128bGlobal, dim3(1), dim3(n_threads),
                       kernel_args, 0, stream);
      cudaStreamSynchronize(stream);
      EXPECT_GPU_CPU_VEC_EQ((int *)args.input, expected_inputs[i]);
      EXPECT_GPU_CPU_VEC_EQ((int *)args.extra_buff, expected_extra_buff[i]);
    });
  }
  for (auto &t : ths) {
    t.join();
  }
}

int *CreateRandomArray(int size, int range) {
  int *ret = new int[size];
  for (int i = 0; i < size; ++i) {
    ret[i] = rand() % range;
  }
  return ret;
}

int *CreateUniquedArray(int size, int range, std::set<int> *st = nullptr) {
  int *ret = new int[size];
  std::set<int> exists;
  for (int i = 0; i < size; ++i) {
    while (1) {
      int t = rand() % range;
      if (st != nullptr && st->count(t) > 0) continue;
      if (exists.count(t) > 0) continue;
      exists.insert(t);
      ret[i] = t;
      break;
    }
  }
  std::sort(ret, ret + size);
  return ret;
}

void BuildLargeCommPatternInfo(
    std::vector<CommPatternInfo> *infos,
    std::vector<CommPatternInfo> *cpu_infos, std::vector<int *> *inputs,
    std::vector<int *> *cpu_inputs, int n_peers,
    const std::vector<std::pair<int, int>> &send_recv_size, int input_size,
    int extra_buff_size, int feat_size, int buff_size) {
  std::vector<RingCommPatternInfo *> ring_infos;
  for (auto &info : *cpu_infos) {
    ring_infos.push_back(info.GetRingCommPatternInfo());
  }
  for (int i = 0; i < n_peers; ++i) {
    cpu_inputs->at(i) = CreateRandomArray(input_size * feat_size, 100);
    ring_infos[i]->dev_extra_mem =
        CreateRandomArray(extra_buff_size * feat_size, 100);
    int send_size = send_recv_size[i].first;
    int recv_size = send_recv_size[i].second;
    ring_infos[i]->send_ids =
        CreateUniquedArray(send_size, input_size + extra_buff_size);
    std::set<int> id_set;
    for (int j = 0; j < send_size; ++j) {
      auto &id = ring_infos[i]->send_ids[j];
      id_set.insert(id);
      if (id >= input_size) {
        id = ENCODE(id - input_size);
      }
    }
    ring_infos[i]->recv_ids =
        CreateUniquedArray(recv_size, input_size + extra_buff_size, &id_set);
    for (int j = 0; j < recv_size; ++j) {
      auto &id = ring_infos[i]->recv_ids[j];
      if (id >= input_size) {
        id = ENCODE(id - input_size);
        int t = ENCODE(id);
        for (int k = 0; k < feat_size; ++k) {
          ((int *)ring_infos[i]->dev_extra_mem)[t * feat_size + k] = -1;
        }
      } else {
        for (int k = 0; k < feat_size; ++k) {
          cpu_inputs->at(i)[id * feat_size + k] = -1;
        }
      }
    }
    GCCLMallocAndCopy(&inputs->at(i), cpu_inputs->at(i),
                      input_size * feat_size);
    GCCLMallocAndCopy(
        (int **)&infos->at(i).GetRingCommPatternInfo()->dev_extra_mem,
        (int *)ring_infos[i]->dev_extra_mem, input_size * feat_size);
    GCCLMallocAndCopy(&infos->at(i).GetRingCommPatternInfo()->send_ids,
                      ring_infos[i]->send_ids, send_size);
    GCCLMallocAndCopy(&infos->at(i).GetRingCommPatternInfo()->recv_ids,
                      ring_infos[i]->recv_ids, recv_size);
    int dev_mem_size = offsetof(RecvDevMem, buff) + buff_size;
    GCCLCudaMalloc((char **)&infos->at(i)
                       .GetRingCommPatternInfo()
                       ->forward_conn.recv_dev_mem,
                   dev_mem_size);
    GCCLCudaMalloc(
        &infos->at(i).GetRingCommPatternInfo()->forward_conn.send_dev_mem, 1);
  }
}

void ConnectOnRing(std::vector<CommPatternInfo> *infos, int n_peers) {
  for (int i = 0; i < n_peers; ++i) {
    int next = (i + 1) % n_peers;
    int prev = (i + n_peers - 1) % n_peers;
    SendDevMem *prev_send_mem, *my_send_mem;
    RecvDevMem *next_recv_mem, *my_recv_mem;

    next_recv_mem =
        infos->at(next).GetRingCommPatternInfo()->forward_conn.recv_dev_mem;
    my_send_mem =
        infos->at(i).GetRingCommPatternInfo()->forward_conn.send_dev_mem;
    my_recv_mem =
        infos->at(i).GetRingCommPatternInfo()->forward_conn.recv_dev_mem;
    prev_send_mem =
        infos->at(prev).GetRingCommPatternInfo()->forward_conn.send_dev_mem;

    infos->at(i)
        .GetRingCommPatternInfo()
        ->forward_conn.conn_info.my_substage_ready =
        &my_recv_mem->substage_ready;
    infos->at(i)
        .GetRingCommPatternInfo()
        ->forward_conn.conn_info.my_substage_done = &my_send_mem->substage_done;

    infos->at(i)
        .GetRingCommPatternInfo()
        ->forward_conn.conn_info.next_substage_ready =
        &next_recv_mem->substage_ready;
    infos->at(i)
        .GetRingCommPatternInfo()
        ->forward_conn.conn_info.prev_substage_done =
        &prev_send_mem->substage_done;
    infos->at(i)
        .GetRingCommPatternInfo()
        ->forward_conn.conn_info.next_recv_buff = &next_recv_mem->buff;
  }
}

void GetExpectedResult(std::vector<std::vector<int>> *exp_inputs,
                       std::vector<std::vector<int>> *exp_extra_buff,
                       const std::vector<CommPatternInfo> &cpu_infos,
                       const std::vector<int *> &cpu_inputs, int n_peers,
                       const std::vector<std::pair<int, int>> &send_recv_size,
                       int input_size, int extra_buff_size, int feat_size) {
  for (int i = 0; i < n_peers; ++i) {
    int prev = (i + n_peers - 1) % n_peers;
    exp_inputs->push_back(std::vector<int>(
        cpu_inputs[i], cpu_inputs[i] + input_size * feat_size));
    exp_extra_buff->push_back(std::vector<int>(
        (int *)cpu_infos[i].GetRingCommPatternInfo()->dev_extra_mem,
        (int *)cpu_infos[i].GetRingCommPatternInfo()->dev_extra_mem +
            extra_buff_size * feat_size));
    int prev_send_size = send_recv_size[prev].first;
    int recv_size = send_recv_size[i].second;
    CHECK(prev_send_size == recv_size);
    for (int j = 0; j < recv_size; ++j) {
      int send_id = cpu_infos[prev].GetRingCommPatternInfo()->send_ids[j];
      int recv_id = cpu_infos[i].GetRingCommPatternInfo()->recv_ids[j];
      int *send_ptr, *val_ptr;
      if (send_id < 0) {
        send_id = ENCODE(send_id);
        send_ptr =
            (int *)cpu_infos[prev].GetRingCommPatternInfo()->dev_extra_mem +
            send_id * feat_size;
      } else {
        send_ptr = cpu_inputs[prev] + send_id * feat_size;
      }
      if (recv_id < 0) {
        recv_id = ENCODE(recv_id);
        val_ptr = exp_extra_buff->at(i).data() + recv_id * feat_size;
      } else {
        val_ptr = exp_inputs->at(i).data() + recv_id * feat_size;
      }
      for (int k = 0; k < feat_size; ++k) {
        *(val_ptr + k) = *(send_ptr + k);
      }
    }
  }
}

TEST_F(TestPrimitives, Copy128bLarge) {
  int buff_size = 128;  // bytes
  int n_peers = 3;
  int n_threads = 4;
  int feat_size = 4;
  int record_size = feat_size * sizeof(int);
  int input_size = 1024, extra_buff_size = 1024;  // n elements
  std::vector<CommPatternInfo> infos(3), cpu_infos(3);
  std::vector<int *> inputs(n_peers), cpu_inputs(n_peers);

  std::vector<std::pair<int, int>> send_recv_size = {
      {1280, 256}, {16, 1280}, {256, 16}};
  // std::vector<std::pair<int, int>> send_recv_size = {
  //     {4, 2}, {2, 4}, {2, 2}};
  cudaSetDevice(0);
  // Element type is int
  BuildLargeCommPatternInfo(&infos, &cpu_infos, &inputs, &cpu_inputs, n_peers,
                            send_recv_size, input_size, extra_buff_size,
                            feat_size, buff_size);
  ConnectOnRing(&infos, n_peers);

  std::vector<std::vector<int>> exp_inputs, exp_extra_buff;

  GetExpectedResult(&exp_inputs, &exp_extra_buff, cpu_infos, cpu_inputs,
                    n_peers, send_recv_size, input_size, extra_buff_size,
                    feat_size);
  // for(int i = 0; i < n_peers; ++i) {
  //	printf("Send id for %d: %s\n", i,
  // VecToString(std::vector<int>(cpu_infos[i].send_ids, cpu_infos[i].send_ids +
  // send_recv_size[i].first)).c_str());
  //	printf("Recv id for %d: %s\n", i,
  // VecToString(std::vector<int>(cpu_infos[i].recv_ids, cpu_infos[i].recv_ids +
  // send_recv_size[i].second)).c_str());
  //	printf("Expected input for      %d: %s\n", i,
  // VecToString(exp_inputs[i]).c_str());
  //	printf("Expected extra buff for %d: %s\n", i,
  // VecToString(exp_extra_buff[i]).c_str());
  //}
  std::vector<RingCommPatternInfo *> ring_infos;
  for (auto &info : infos) {
    ring_infos.push_back(info.GetRingCommPatternInfo());
  }

  std::vector<std::thread> ths;
  for (int i = 0; i < n_peers; ++i) {
    ths.emplace_back([&ring_infos, i, record_size, buff_size, &send_recv_size,
                      &inputs, &exp_inputs, n_threads, &exp_extra_buff]() {
      CopyArgs args(-1, -1,
                    ring_infos[i]->forward_conn.conn_info.my_substage_ready,
                    ring_infos[i]->forward_conn.conn_info.my_substage_done,
                    ring_infos[i]->forward_conn.conn_info.next_substage_ready,
                    ring_infos[i]->forward_conn.conn_info.prev_substage_done);
      cudaSetDevice(0);
      args.n_128b = record_size / PACK_SIZE;
      args.buff_n_128b = buff_size / PACK_SIZE;
      args.input = (Pack128 *)inputs[i];
      args.recv_buff =
          (Pack128 *)&ring_infos[i]->forward_conn.recv_dev_mem->buff;
      args.extra_buff = (Pack128 *)ring_infos[i]->dev_extra_mem;
      args.next_recv_buff =
          (Pack128 *)ring_infos[i]->forward_conn.conn_info.next_recv_buff;
      args.send_ids = ring_infos[i]->send_ids;
      args.send_size = send_recv_size[i].first;
      args.recv_ids = ring_infos[i]->recv_ids;
      args.recv_size = send_recv_size[i].second;
      args.extra_buff_size = exp_extra_buff[i].size();
      args.max_comm_size = 1280;
      void *kernel_args[] = {&args};
      cudaStream_t stream;
      cudaStreamCreate(&stream);
      cudaLaunchKernel((void *)Copy128bGlobal, dim3(1), dim3(n_threads),
                       kernel_args, 0, stream);
      cudaStreamSynchronize(stream);
      // LOG(INFO) << "Result input for " << i << " is:" <<
      // CudaVecToString((int*)args.input, exp_inputs[i].size());
      // LOG(INFO) << "Result buff  for " << i << " is:" <<
      // CudaVecToString((int*)args.extra_buff, exp_extra_buff[i].size());
      EXPECT_GPU_CPU_VEC_EQ((int *)args.input, exp_inputs[i]);
      EXPECT_GPU_CPU_VEC_EQ((int *)args.extra_buff, exp_extra_buff[i]);
    });
  }
  for (auto &t : ths) {
    t.join();
  }
}

}  // namespace

}  // namespace gccl
