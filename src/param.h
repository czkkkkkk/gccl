#pragma once

#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>

#include "glog/logging.h"

namespace gccl {

template <typename T>
T GetEnvParam(const std::string &key, T default_value) {
  auto gccl_str = std::string("GCCL_") + key;
  char *ptr = std::getenv(gccl_str.c_str());
  if (ptr == nullptr) return default_value;
  std::stringstream converter(ptr);
  T ret;
  converter >> ret;
  return ret;
}

template <typename T>
T GetEnvParam(const char *str, T default_value) {
  return GetEnvParam<T>(std::string(str), default_value);
}

#define GCCL_PARAM(name, env, default_value)                              \
  pthread_mutex_t gcclParamMutex##name = PTHREAD_MUTEX_INITIALIZER;       \
  int64_t gcclParam##name() {                                             \
    static_assert(default_value != -1LL, "default value cannot be -1");   \
    static int64_t value = -1LL;                                          \
    pthread_mutex_lock(&gcclParamMutex##name);                            \
    if (value == -1LL) {                                                  \
      value = default_value;                                              \
      char *str = getenv("GCCL_" env);                                    \
      if (str && strlen(str) > 0) {                                       \
        errno = 0;                                                        \
        int64_t v = strtoll(str, NULL, 0);                                \
        if (errno) {                                                      \
          LOG(INFO) << "Invalid value " << str << " for "                 \
                    << "GCCL_" env << ", using default value " << value;  \
        } else {                                                          \
          value = v;                                                      \
          LOG(INFO) << "GCCL_" env << " set by environment to " << value; \
        }                                                                 \
      }                                                                   \
    }                                                                     \
    pthread_mutex_unlock(&gcclParamMutex##name);                          \
    return value;                                                         \
  }

}  // namespace gccl