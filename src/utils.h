#pragma once

#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <random>
#include <string>

#include "glog/logging.h"

#include "gccl.h"

namespace gccl {

#define SYSCHECK(call, name)                                     \
  do {                                                           \
    int ret = -1;                                                \
    while (ret == -1) {                                          \
      SYSCHECKVAL(call, name, ret);                              \
      if (ret == -1) {                                           \
        LOG(ERROR) << "Got " << strerror(errno) << ", retrying"; \
      }                                                          \
    }                                                            \
  } while (0);

#define SYSCHECKVAL(call, name, retval)                                    \
  do {                                                                     \
    retval = call;                                                         \
    if (retval == -1 && errno != EINTR && errno != EWOULDBLOCK &&          \
        errno != EAGAIN) {                                                 \
      LOG(ERROR) << "Call to " << name << " failed : " << strerror(errno); \
      return gcclSystemError;                                              \
    }                                                                      \
  } while (0);

#define SYSCHECKNTIMES(call, name, times, usec, exptype)                    \
  do {                                                                      \
    int ret = -1;                                                           \
    int count = 0;                                                          \
    while (ret == -1 && count < times) {                                    \
      SYSCHECKVALEXP(call, name, ret, exptype);                             \
      count++;                                                              \
      if (ret == -1) {                                                      \
        usleep(usec);                                                       \
      }                                                                     \
    }                                                                       \
    if (ret == -1) {                                                        \
      LOG(ERROR) << "Call to " << name << " timeout : " << strerror(errno); \
      return gcclSystemError;                                               \
    }                                                                       \
  } while (0);

#define SYSCHECKVALEXP(call, name, retval, exptype)                        \
  do {                                                                     \
    retval = call;                                                         \
    if (retval == -1 && errno != EINTR && errno != EWOULDBLOCK &&          \
        errno != EAGAIN && errno != exptype) {                             \
      LOG(ERROR) << "Call to " << name << " failed : " << strerror(errno); \
      return gcclSystemError;                                              \
    }                                                                      \
  } while (0);

int GetAvailablePort();
std::string GetHostName();
inline bool CheckDirExists(const std::string &dirname) {
  DIR *dir = opendir(dirname.c_str());
  if (dir) {
    closedir(dir);
    return true;
  }
  return false;
}
inline void CreateDir(const std::string &dirname) {
  int result = mkdir(dirname.c_str(), 0777);
  CHECK(result == 0) << "Cannot create directory on " << dirname;
}

void getHostName(char *hostname, int maxlen);
uint64_t getHostHash();
uint64_t getPidHash();

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char *string, struct netIf *ifList, int maxList);
bool matchIfList(const char *string, int port, struct netIf *ifList,
                 int listSize);

template <typename T>
void UniqueVec(std::vector<T> &vec) {
  std::sort(vec.begin(), vec.end());
  vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
}

template <typename T>
std::string VecToString(const std::vector<T> &vec, char sep = ' ') {
  std::string ret;
  for (int i = 0; i < vec.size(); ++i) {
    ret += std::to_string(vec[i]);
    if (i + 1 < vec.size()) {
      ret += sep;
    }
  }
  return ret;
}

template <typename T>
std::string VecToString(const T *ptr, int size, char sep = ' ') {
  return VecToString(std::vector<T>(ptr, ptr + size), sep);
}

template <typename T>
void CopyVectorToRawPtr(T **rp, const std::vector<T> &vec) {
  *rp = new T[vec.size()];
  memcpy((void *)*rp, (void *)vec.data(), vec.size() * sizeof(T));
}

template <typename T>
std::vector<std::vector<std::vector<T>>> Build3DVector(int d0, int d1) {
  return std::vector<std::vector<std::vector<T>>>(
      d0, std::vector<std::vector<T>>(d1));
}

template <typename T>
std::vector<T> Repeat(const std::vector<T> &vec, int t) {
  std::vector<T> ret(vec.size() * t);
  for (int i = 0; i < vec.size(); ++i) {
    for (int j = 0; j < t; ++j) {
      ret[i * t + j] = vec[i];
    }
  }
  return ret;
}

using TP = std::chrono::steady_clock::time_point;

static inline TP GetTime() { return std::chrono::steady_clock::now(); }

static inline long int TimeDiff(const TP &begin, const TP &end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
      .count();
}

class RandomGenerator {
 public:
  typedef unsigned long long KEY;
  static constexpr KEY BASE = 1000009;
  static constexpr KEY MOD = 1000000007;
  static constexpr KEY INC = 2333333;
  RandomGenerator(KEY seed) : seed_(seed) {}
  KEY Rand() {
    seed_ = (seed_ * BASE + INC) % MOD;
    return seed_;
  }

 private:
  KEY seed_;
};

}  // namespace gccl
