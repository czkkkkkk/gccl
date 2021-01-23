#pragma once

#include <cassert>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "base/macros.h"

namespace gccl {

class BinStream {
 public:
  BinStream();
  explicit BinStream(size_t sz);
  explicit BinStream(const std::vector<char> &v);
  explicit BinStream(std::vector<char> &&v);
  BinStream(const char *src, size_t sz);
  BinStream(const BinStream &stream);
  BinStream(BinStream &&stream);
  virtual ~BinStream();

  BinStream &operator=(BinStream &&stream);

  size_t hash();
  void clear();
  void purge();
  void resize(size_t size);
  void seek(size_t pos);

  void append(const BinStream &m);
  void push_back_bytes(const char *src, size_t sz);
  virtual void *pop_front_bytes(size_t sz);
  virtual size_t size() const { return buffer_.size() - front_; }

  /// Note that this method just returns the pointer pointing to the very
  /// beginning of the buffer_, and doesn't care about how much data have
  /// been read.
  inline char *get_buffer() { return &buffer_[0]; }
  inline const std::vector<char> &get_buffer_vector() { return buffer_; }
  inline const char *get_remained_buffer() const {
    return (&buffer_[0]) + front_;
  }
  inline std::string to_string() const {
    return std::string(buffer_.begin() + front_, buffer_.end());
  }
  inline size_t get_total_size() const { return buffer_.size(); }

 protected:
  std::vector<char> buffer_;

 private:
  size_t front_;
};

template <typename T>
class has_serialize {
 private:
  template <typename U>
  static constexpr auto check(int) -> typename std::is_same<
      decltype(std::declval<U>().serialize(*(new BinStream))),
      BinStream &>::type;

  template <typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<T>(0)) type;

 public:
  static constexpr bool value = type::value;
};

template <typename T>
class has_deserialize {
 private:
  template <typename U>
  static constexpr auto check(int) -> typename std::is_same<
      decltype(std::declval<U>().deserialize(*(new BinStream))),
      BinStream &>::type;

  template <typename>
  static constexpr std::false_type check(...);

  typedef decltype(check<T>(0)) type;

 public:
  static constexpr bool value = type::value;
};

template <typename InputT>
typename std::enable_if<has_serialize<InputT>::value, BinStream>::type &
operator<<(BinStream &stream, const InputT &x) {
  x.serialize(stream);
  return stream;
}

template <typename OutputT>
typename std::enable_if<has_deserialize<OutputT>::value, BinStream>::type &
operator>>(BinStream &stream, OutputT &x) {
  x.deserialize(stream);
  return stream;
}

template <typename InputT>
typename std::enable_if<!has_serialize<InputT>::value, BinStream>::type &
operator<<(BinStream &stream, const InputT &x) {
  static_assert(
      IS_TRIVIALLY_COPYABLE(InputT),
      "For non trivially copyable type, serialization functions are needed");
  stream.push_back_bytes((char *)&x, sizeof(InputT));
  return stream;
}

template <typename OutputT>
typename std::enable_if<!has_deserialize<OutputT>::value, BinStream>::type &
operator>>(BinStream &stream, OutputT &x) {
  static_assert(
      IS_TRIVIALLY_COPYABLE(OutputT),
      "For non trivially copyable type, serialization functions are needed");
  x = *(OutputT *)(stream.pop_front_bytes(sizeof(OutputT)));
  return stream;
}

template <typename InputT>
BinStream &operator<<(BinStream &stream, const std::vector<InputT> &v) {
  size_t len = v.size();
  stream << len;
  for (int i = 0; i < v.size(); ++i) stream << v[i];
  return stream;
}

template <typename OutputT>
BinStream &operator>>(BinStream &stream, std::vector<OutputT> &v) {
  size_t len;
  stream >> len;
  v.clear();
  v.reserve(len);
  for (int i = 0; i < len; ++i) {
    OutputT elem;
    stream >> elem;
    v.push_back(std::move(elem));
  }
  return stream;
}

template <typename K, typename V>
BinStream &operator<<(BinStream &stream, const std::map<K, V> &map) {
  size_t len = map.size();
  stream << len;
  for (auto &elem : map) stream << elem;
  return stream;
}

template <typename K, typename V>
BinStream &operator>>(BinStream &stream, std::map<K, V> &map) {
  size_t len;
  stream >> len;
  map.clear();
  for (int i = 0; i < len; i++) {
    std::pair<K, V> elem;
    stream >> elem;
    map.insert(std::move(elem));
  }
  return stream;
}

template <typename K>
BinStream &operator<<(BinStream &stream, const std::set<K> &set) {
  size_t len = set.size();
  stream << len;
  for (auto &elem : set) stream << elem;
  return stream;
}

template <typename K>
BinStream &operator>>(BinStream &stream, std::set<K> &set) {
  size_t len;
  stream >> len;
  set.clear();
  for (int i = 0; i < len; ++i) {
    K elem;
    stream >> elem;
    set.insert(std::move(elem));
  }
  return stream;
}

template <typename K, typename HashT>
BinStream &operator<<(BinStream &stream,
                      const std::unordered_set<K, HashT> &unordered_set) {
  size_t len = unordered_set.size();
  stream << len;
  for (auto &elem : unordered_set) stream << elem;
  return stream;
}

template <typename K, typename HashT>
BinStream &operator>>(BinStream &stream,
                      std::unordered_set<K, HashT> &unordered_set) {
  size_t len;
  stream >> len;
  unordered_set.clear();
  for (int i = 0; i < len; ++i) {
    K elem;
    stream >> elem;
    unordered_set.insert(std::move(elem));
  }
  return stream;
}

template <typename K, typename V, typename HashT>
BinStream &operator<<(BinStream &stream,
                      const std::unordered_map<K, V, HashT> &unordered_map) {
  size_t len = unordered_map.size();
  stream << len;
  for (auto &elem : unordered_map) stream << elem;
  return stream;
}

template <typename K, typename V, typename HashT>
BinStream &operator>>(BinStream &stream,
                      std::unordered_map<K, V, HashT> &unordered_map) {
  size_t len;
  stream >> len;
  unordered_map.clear();
  for (int i = 0; i < len; i++) {
    std::pair<K, V> elem;
    stream >> elem;
    unordered_map.insert(std::move(elem));
  }
  return stream;
}

template <typename T>
BinStream &operator<<(BinStream &stream, const std::shared_ptr<T> &ptr) {
  stream << *ptr;
  return stream;
}

template <typename T>
BinStream &operator>>(BinStream &stream, std::shared_ptr<T> &ptr) {
  T tmp;
  stream >> tmp;
  ptr.reset(new T);
  *ptr = tmp;
  return stream;
}

template <typename T>
BinStream &operator<<(BinStream &stream, const std::unique_ptr<T> &ptr) {
  stream << *ptr;
  return stream;
}

template <typename T>
BinStream &operator>>(BinStream &stream, std::unique_ptr<T> &ptr) {
  T tmp;
  stream >> tmp;
  ptr.reset(new T);
  *ptr = tmp;
  return stream;
}

template <typename InputT>
BinStream &operator<<(BinStream &stream, const std::basic_string<InputT> &v) {
  size_t len = v.size();
  stream << len;
  for (auto &elem : v) stream << elem;
  return stream;
}

template <typename OutputT>
BinStream &operator>>(BinStream &stream, std::basic_string<OutputT> &v) {
  size_t len;
  stream >> len;
  v.clear();
  try {
    v.resize(len);
  } catch (std::exception e) {
    assert(false);
  }
  for (auto &elem : v) stream >> elem;
  return stream;
}

template <typename FirstT, typename SecondT>
BinStream &operator<<(BinStream &stream, const std::pair<FirstT, SecondT> &p) {
  stream << p.first << p.second;
  return stream;
}

template <typename FirstT, typename SecondT>
BinStream &operator>>(BinStream &stream, std::pair<FirstT, SecondT> &p) {
  stream >> p.first >> p.second;
  return stream;
}

BinStream &operator<<(BinStream &stream, const BinStream &bin);
BinStream &operator>>(BinStream &stream, BinStream &bin);

BinStream &operator<<(BinStream &stream, const std::string &x);
BinStream &operator>>(BinStream &stream, std::string &x);

BinStream &operator<<(BinStream &stream, const std::vector<bool> &v);
BinStream &operator>>(BinStream &stream, std::vector<bool> &v);

template <typename Value>
Value deser(BinStream &in) {
  Value v;
  in >> v;
  return v;
}

}  // namespace gccl