/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <gsl/pointers>

#include "core/common/common.h"
#include "core/platform/env_time.h"

#ifndef _WIN32
#include <sys/types.h>
#include <unistd.h>
#endif

namespace onnxruntime {

class Thread;

struct ThreadOptions;
#ifdef _WIN32
using PIDType = unsigned long;
#else
using PIDType = pid_t;
#endif

/// \brief An interface used by the onnxruntime implementation to
/// access operating system functionality like the filesystem etc.
///
/// Callers may wish to provide a custom Env object to get fine grain
/// control.
///
/// All Env implementations are safe for concurrent access from
/// multiple threads without any external synchronization.
class Env {
 public:
  virtual ~Env() = default;
  /// for use with Eigen::ThreadPool
  using EnvThread = Thread;

  /// for use with Eigen::ThreadPool
  struct Task {
    std::function<void()> f;
  };
  /// \brief Returns a default environment suitable for the current operating
  /// system.
  ///
  /// Sophisticated users may wish to provide their own Env
  /// implementation instead of relying on this default environment.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static const Env& Default();

  virtual int GetNumCpuCores() const = 0;

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  virtual uint64_t NowMicros() const { return env_time_->NowMicros(); }

  /// \brief Returns the number of seconds since the Unix epoch.
  virtual uint64_t NowSeconds() const { return env_time_->NowSeconds(); }

  /// Sleeps/delays the thread for the prescribed number of micro-seconds.
  /// On Windows, it's the min time to sleep, not the actual one.
  virtual void SleepForMicroseconds(int64_t micros) const = 0;

  /// for use with Eigen::ThreadPool
  virtual EnvThread* CreateThread(std::function<void()> f) const = 0;
  /// for use with Eigen::ThreadPool
  virtual Task CreateTask(std::function<void()> f) const = 0;
  /// for use with Eigen::ThreadPool
  virtual void ExecuteTask(const Task& t) const = 0;

  /// \brief Returns a new thread that is running fn() and is identified
  /// (for debugging/performance-analysis) by "name".
  ///
  /// Caller takes ownership of the result and must delete it eventually
  /// (the deletion will block until fn() stops running).
  virtual Thread* StartThread(const ThreadOptions& thread_options,
                              const std::string& name,
                              std::function<void()> fn) const = 0;
  virtual common::Status FileExists(const char* fname) const = 0;
#ifdef _WIN32
  virtual common::Status FileExists(const wchar_t* fname) const = 0;
#endif
  /// File size must less than 2GB.
  /// No support for non-regular files(e.g. socket, pipe, "/proc/*")
  virtual common::Status ReadFileAsString(const char* fname, std::string* out) const = 0;
#ifdef _WIN32
  virtual common::Status ReadFileAsString(const wchar_t* fname, std::string* out) const = 0;
#endif

#ifdef _WIN32
  //Mainly for use with protobuf library
  virtual common::Status FileOpenRd(const std::wstring& path, /*out*/ gsl::not_null<int*> p_fd) const = 0;
  //Mainly for use with protobuf library
  virtual common::Status FileOpenWr(const std::wstring& path, /*out*/ gsl::not_null<int*> p_fd) const = 0;
#endif
  //Mainly for use with protobuf library
  virtual common::Status FileOpenRd(const std::string& path, /*out*/ gsl::not_null<int*> p_fd) const = 0;
  //Mainly for use with protobuf library
  virtual common::Status FileOpenWr(const std::string& path, /*out*/ gsl::not_null<int*> p_fd) const = 0;
  //Mainly for use with protobuf library
  virtual common::Status FileClose(int fd) const = 0;
  //This functions is always successful. It can't fail.
  virtual PIDType GetSelfPid() const = 0;

  // \brief Load a dynamic library.
  //
  // Pass "library_filename" to a platform-specific mechanism for dynamically
  // loading a library.  The rules for determining the exact location of the
  // library are platform-specific and are not documented here.
  //
  // On success, returns a handle to the library in "*handle" and returns
  // OK from the function.
  // Otherwise returns nullptr in "*handle" and an error status from the
  // function.
  // TODO(@chasun): rename LoadLibrary to something else. LoadLibrary is already defined in Windows.h
  virtual common::Status LoadLibrary(const std::string& library_filename, void** handle) const = 0;

  virtual common::Status UnloadLibrary(void* handle) const = 0;

  // \brief Get a pointer to a symbol from a dynamic library.
  //
  // "handle" should be a pointer returned from a previous call to LoadLibrary.
  // On success, store a pointer to the located symbol in "*symbol" and return
  // OK from the function. Otherwise, returns nullptr in "*symbol" and an error
  // status from the function.
  virtual common::Status GetSymbolFromLibrary(void* handle, const std::string& symbol_name, void** symbol) const = 0;

  // \brief build the name of dynamic library.
  //
  // "name" should be name of the library.
  // "version" should be the version of the library or NULL
  // returns the name that LoadLibrary() can use
  virtual std::string FormatLibraryFileName(const std::string& name, const std::string& version) const = 0;

 protected:
  Env();

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(Env);
  EnvTime* env_time_ = EnvTime::Default();
};

/// Represents a thread used to run a onnxruntime function.
class Thread {
 public:
  Thread() noexcept = default;

  /// Blocks until the thread of control stops running.
  virtual ~Thread();

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(Thread);
};

/// \brief Options to configure a Thread.
///
/// Note that the options are all hints, and the
/// underlying implementation may choose to ignore it.
struct ThreadOptions {
  /// Thread stack size to use (in bytes).
  size_t stack_size = 0;  // 0: use system default value
  /// Guard area size to use near thread stacks to use (in bytes)
  size_t guard_size = 0;  // 0: use system default value
};

}  // namespace onnxruntime
