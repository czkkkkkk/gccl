#pragma once

#include <map>
#include <thread>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "gtest/gtest.h"

namespace gccl {

template <typename Lambda>
void MultiThreading(int n_threads, Lambda lambda) {
  std::vector<std::thread> ths;
  for (int i = 0; i < n_threads; ++i) {
    ths.emplace_back(lambda, i);
  }
  for (auto &t : ths) t.join();
}

template <typename T>
typename std::enable_if<std::is_same<T, float>::value>::type SEPERATE_EXPECT_EQ(
    const T &lhs, const T &rhs) {
  EXPECT_FLOAT_EQ(lhs, rhs);
}

template <typename T>
typename std::enable_if<!std::is_same<T, float>::value>::type
SEPERATE_EXPECT_EQ(const T &lhs, const T &rhs) {
  EXPECT_EQ(lhs, rhs);
}

template <typename T>
void EXPECT_VEC_EQ(const std::vector<T> &lhs, const std::vector<T> &rhs) {
  EXPECT_EQ(lhs.size(), rhs.size());
  for (int i = 0; i < lhs.size(); ++i) {
    SEPERATE_EXPECT_EQ(lhs[i], rhs[i]);
  }
}

template <typename T>
void EXPECT_VEC_EQ(const T *start, const T *end, const std::vector<T> &rhs) {
  std::vector<T> lhs(start, end);
  EXPECT_VEC_EQ(lhs, rhs);
}

template <typename T>
void EXPECT_VEC_EQ(const T *start, const std::vector<T> &rhs) {
  std::vector<T> lhs(start, start + rhs.size());
  EXPECT_VEC_EQ(lhs, rhs);
}

template <typename T>
void EXPECT_GPU_CPU_VEC_EQ(const T *ptr, const std::vector<T> &rhs) {
  std::vector<T> lhs(rhs.size());
  cudaMemcpy(lhs.data(), ptr, sizeof(T) * rhs.size(), cudaMemcpyDeviceToHost);
  EXPECT_VEC_EQ(lhs, rhs);
}

template <typename KType, typename VType>
void EXPECT_MAP_EQ(const std::map<KType, VType> &lhs,
                   const std::map<KType, VType> &rhs) {
  EXPECT_EQ(lhs.size(), rhs.size());
  for (const auto &pair : rhs) {
    auto k = pair.first;
    auto v = pair.second;
    EXPECT_TRUE(lhs.count(k) > 0 && lhs.at(k) == v);
  }
}

}  // namespace gccl
