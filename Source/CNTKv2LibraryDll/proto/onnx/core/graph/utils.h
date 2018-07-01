#pragma once

#include <fcntl.h>
#include <fstream>
#ifdef _WIN32
#include <io.h>
#else
#include <sys/io.h>
#include <unistd.h>
#endif

#include <mutex>
#include <string>
#include <unordered_map>
#include "core/common/common.h"
#include "core/common/status.h"
#include "onnx/onnx_pb.h"

// #include "gsl/pointers"

namespace Lotus{
using namespace ::Lotus::Common;
#ifdef _WIN32
inline Status FileOpenRd(const std::wstring& path, /*out*/ int* p_fd) {
  _wsopen_s(p_fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
  if (0 > *p_fd) {
    return Status(SYSTEM, errno);
  }
  return Status::OK();
}

inline Status FileOpenWr(const std::wstring& path, /*out*/ int* p_fd) {
  _wsopen_s(p_fd, path.c_str(), _O_CREAT | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
  if (0 > *p_fd) {
    return Status(SYSTEM, errno);
  }
  return Status::OK();
}
#endif

inline Status FileOpenRd(const std::string& path, /*out*/ int* p_fd) {
#ifdef _WIN32
  _sopen_s(p_fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#else
  *p_fd = open(path.c_str(), O_RDONLY);
#endif
  if (0 > *p_fd) {
    return Status(SYSTEM, errno);
  }
  return Status::OK();
}

inline Status FileOpenWr(const std::string& path, /*out*/ int* p_fd) {
#ifdef _WIN32
  _sopen_s(p_fd, path.c_str(), _O_CREAT | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#else
  *p_fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif
  if (0 > *p_fd) {
    return Status(SYSTEM, errno);
  }
  return Status::OK();
}

inline Status FileClose(int fd) {
  int ret = 0;
#ifdef _WIN32
  ret = _close(fd);
#else
  ret = close(fd);
#endif
  if (0 != ret) {
    return Status(SYSTEM, errno);
  }
  return Status::OK();
}
}  // namespace Lotus
