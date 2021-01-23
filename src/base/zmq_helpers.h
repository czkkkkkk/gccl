#pragma once

#include <string>

#include "glog/logging.h"
#include "zmq.h"
#include "zmq.hpp"

#include "base/bin_stream.h"

namespace gccl {

// Define blocking constant since it does not exist in zmq.hpp.
// ZMQ_NOBLOCK = ZMQ_DONTWAIT = 1 -> ZMQ_BLOCKING = !ZMQ_NOBLOCK
const int ZMQ_BLOCKING = !ZMQ_NOBLOCK;

// ZMQ send part.

inline void zmq_send_common(zmq::socket_t *socket, const void *data,
                            const size_t &len, int flag = ZMQ_BLOCKING) {
  CHECK(socket != nullptr) << "[ZMQ Helper] zmq::socket_t cannot be nullptr!";
  CHECK(data != nullptr || len == 0)
      << "[ZMQ Helper] data and len are not matched!";
  while (true) try {
      size_t bytes = socket->send(data, len, flag);
      CHECK(bytes == len) << "[ZMQ Helper] zmq::send error!";
      break;
    } catch (zmq::error_t e) {
      switch (e.num()) {
        case EHOSTUNREACH:
        case EINTR:
          continue;
        default:
          CHECK(false) << "[ZMQ Helper] Invalid type of zmq::error";
      }
    }
}

inline void zmq_send_dummy(zmq::socket_t *socket, int flag = ZMQ_BLOCKING) {
  zmq_send_common(socket, nullptr, 0, flag);
}

inline void zmq_sendmore_dummy(zmq::socket_t *socket) {
  zmq_send_dummy(socket, ZMQ_SNDMORE);
}

inline void zmq_send_int32(zmq::socket_t *socket, int32_t data,
                           int flag = ZMQ_BLOCKING) {
  zmq_send_common(socket, &data, sizeof(int32_t), flag);
}

inline void zmq_sendmore_int32(zmq::socket_t *socket, int32_t data) {
  zmq_send_int32(socket, data, ZMQ_SNDMORE);
}

inline void zmq_send_int64(zmq::socket_t *socket, int64_t data,
                           int flag = ZMQ_BLOCKING) {
  zmq_send_common(socket, &data, sizeof(int64_t), flag);
}

inline void zmq_sendmore_int64(zmq::socket_t *socket, int64_t data) {
  zmq_send_int64(socket, data, ZMQ_SNDMORE);
}

inline void zmq_send_string(zmq::socket_t *socket, const std::string &data,
                            int flag = ZMQ_BLOCKING) {
  zmq_send_common(socket, data.data(), data.length(), flag);
}

inline void zmq_sendmore_string(zmq::socket_t *socket,
                                const std::string &data) {
  zmq_send_string(socket, data, ZMQ_SNDMORE);
}

// FIXME(legend): Whether it needs BinStream.get_buffer()?
inline void zmq_send_binstream(zmq::socket_t *socket, const BinStream &stream,
                               int flag = ZMQ_BLOCKING) {
  zmq_send_common(socket, stream.get_remained_buffer(), stream.size(), flag);
}

inline void zmq_send_binstream_ptr(zmq::socket_t *socket,
                                   const std::shared_ptr<BinStream> &stream,
                                   int flag = ZMQ_BLOCKING) {
  CHECK(socket != nullptr) << "[ZMQ Helper] zmq::socket_t cannot be nullptr!";

  const void *data = stream->get_remained_buffer();
  size_t len = stream->size();

  CHECK(data != nullptr || len == 0)
      << "[ZMQ Helper] data and len are not matched!";

  while (true) {
    try {
      zmq_msg_t msg;
      void *hint = new std::shared_ptr<BinStream>(stream);
      int rc = zmq_msg_init_data(&msg, const_cast<void *>(data), len,
                                 [](void *, void *hint) {
                                   delete (std::shared_ptr<BinStream> *)(hint);
                                 },
                                 hint);
      CHECK_EQ(rc, 0) << "[ZMQ Helper] Cannot init data for zmq send";
      rc = zmq_msg_send(&msg, (void *)(*socket), flag);
      CHECK_EQ(rc, len) << "[ZMQ Helper]  zmq::send error! errno "
                        << strerror(errno);
      break;
    } catch (zmq::error_t e) {
      switch (e.num()) {
        case EHOSTUNREACH:
        case EINTR:
          continue;
        default:
          CHECK(false) << "[ZMQ Helper] Invalid type of zmq::error";
      }
    }
  }
}

// ZMQ receive part.

inline void zmq_recv_common(zmq::socket_t *socket, zmq::message_t *msg,
                            int flag = ZMQ_BLOCKING) {
  CHECK(socket != nullptr) << "[ZMQ Helper] zmq::socket_t cannot be nullptr!";
  CHECK(msg != nullptr) << "[ZMQ Helper] zmq::message_t cannot be nullptr!";
  while (true) try {
      bool successful = socket->recv(msg, flag);
      CHECK(successful) << "[ZMQ Helper] zmq::recv error!";
      break;
    } catch (zmq::error_t e) {
      if (e.num() == EINTR) continue;
      CHECK(false) << "[ZMQ Helper] Invalid type of zmq::error!";
    }
}

inline void zmq_recv_dummy(zmq::socket_t *socket, int flag = ZMQ_BLOCKING) {
  zmq::message_t msg;
  zmq_recv_common(socket, &msg, flag);
}

inline int32_t zmq_recv_int32(zmq::socket_t *socket) {
  zmq::message_t msg;
  zmq_recv_common(socket, &msg);
  return *reinterpret_cast<int32_t *>(msg.data());
}

inline int64_t zmq_recv_int64(zmq::socket_t *socket) {
  zmq::message_t msg;
  zmq_recv_common(socket, &msg);
  return *reinterpret_cast<int64_t *>(msg.data());
}

inline std::string zmq_recv_string(zmq::socket_t *socket) {
  zmq::message_t msg;
  zmq_recv_common(socket, &msg);
  return std::string(reinterpret_cast<char *>(msg.data()), msg.size());
}

inline BinStream zmq_recv_binstream(zmq::socket_t *socket,
                                    int flag = ZMQ_BLOCKING) {
  zmq::message_t msg;
  zmq_recv_common(socket, &msg, flag);
  BinStream stream;
  stream.push_back_bytes(reinterpret_cast<char *>(msg.data()), msg.size());
  return stream;
}

}  // namespace gccl