/**
 * Derived from caffe2, need copy right annoucement here.
 */

/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "core/common/code_location.h"
#include "core/common/exceptions.h"
#include "core/common/status.h"

namespace onnxruntime {

using TimePoint = std::chrono::high_resolution_clock::time_point;

// Using statements for common classes that we refer to in lotus very often.
// TODO(Task:137) Remove 'using' statements from header files
using common::Status;

#ifdef _WIN32
#define UNUSED_PARAMETER(x) (x)
#else
#define UNUSED_PARAMETER(x) (void)(x)
#endif

#ifndef LOTUS_HAVE_ATTRIBUTE
#ifdef __has_attribute
#define LOTUS_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define LOTUS_HAVE_ATTRIBUTE(x) 0
#endif
#endif

// LOTUS_ATTRIBUTE_UNUSED
//
// Prevents the compiler from complaining about or optimizing away variables
// that appear unused on Linux
#if LOTUS_HAVE_ATTRIBUTE(unused) || (defined(__GNUC__) && !defined(__clang__))
#undef LOTUS_ATTRIBUTE_UNUSED
#define LOTUS_ATTRIBUTE_UNUSED __attribute__((__unused__))
#else
#define LOTUS_ATTRIBUTE_UNUSED
#endif

// macro to explicitly ignore the return value from a function call so Code Analysis doesn't complain
#define IGNORE_RETURN_VALUE(fn) \
  static_cast<void>(fn)

inline static std::vector<std::string> GetStackTrace() { return {}; }

// __PRETTY_FUNCTION__ isn't a macro on gcc, so use a check for _MSC_VER
// so we only define it as one for MSVC
#if (_MSC_VER && !defined(__PRETTY_FUNCTION__))
#define __PRETTY_FUNCTION__ __FUNCTION__
#endif

// Capture where a message is coming from. Use __FUNCTION__ rather than the much longer __PRETTY_FUNCTION__
#define WHERE \
  ::onnxruntime::CodeLocation(__FILE__, __LINE__, __FUNCTION__)

#define WHERE_WITH_STACK \
  ::onnxruntime::CodeLocation(__FILE__, __LINE__, __PRETTY_FUNCTION__, ::onnxruntime::GetStackTrace())

// Throw an exception with optional message.
// NOTE: The arguments get streamed into a string via ostringstream::operator<<
// DO NOT use a printf format string, as that will not work as you expect.
#define LOTUS_THROW(...) throw ::onnxruntime::LotusException(WHERE_WITH_STACK, ::onnxruntime::MakeString(__VA_ARGS__))

// Just in order to mark things as not implemented. Do not use in final code.
#define LOTUS_NOT_IMPLEMENTED(...) throw ::onnxruntime::NotImplementedException(::onnxruntime::MakeString(__VA_ARGS__))

// Check condition.
// NOTE: The arguments get streamed into a string via ostringstream::operator<<
// DO NOT use a printf format string, as that will not work as you expect.
#define LOTUS_ENFORCE(condition, ...) \
  if (!(condition)) throw ::onnxruntime::LotusException(WHERE_WITH_STACK, #condition, ::onnxruntime::MakeString(__VA_ARGS__))

#define LOTUS_MAKE_STATUS(category, code, ...) \
  ::onnxruntime::common::Status(::onnxruntime::common::category, ::onnxruntime::common::code, ::onnxruntime::MakeString(__VA_ARGS__))

// Check condition. if not met, return status.
#define LOTUS_RETURN_IF_NOT(condition, ...)                                                                                             \
  if (!(condition)) {                                                                                                                   \
    return LOTUS_MAKE_STATUS(LOTUS, FAIL, "Not satsified: " #condition "\n", WHERE.ToString(), ::onnxruntime::MakeString(__VA_ARGS__)); \
  }

// Macros to disable the copy and/or move ctor and assignment methods
// These are usually placed in the private: declarations for a class.

#define LOTUS_DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

#define LOTUS_DISALLOW_ASSIGN(TypeName) TypeName& operator=(const TypeName&) = delete

#define LOTUS_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  LOTUS_DISALLOW_COPY(TypeName);                 \
  LOTUS_DISALLOW_ASSIGN(TypeName)

#define LOTUS_DISALLOW_MOVE(TypeName) \
  TypeName(TypeName&&) = delete;      \
  TypeName& operator=(TypeName&&) = delete

#define LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(TypeName) \
  LOTUS_DISALLOW_COPY_AND_ASSIGN(TypeName);           \
  LOTUS_DISALLOW_MOVE(TypeName)

#define LOTUS_RETURN_IF_ERROR(expr)        \
  do {                                     \
    auto _status = (expr);                 \
    if ((!_status.IsOK())) return _status; \
  } while (0)

// use this macro when cannot early return
#define LOTUS_CHECK_AND_SET_RETVAL(expr) \
  do {                                   \
    if (retval.IsOK()) {                 \
      retval = (expr);                   \
    }                                    \
  } while (0)

// C++ Core Guideline check suppression
#ifdef _MSC_VER
#define GSL_SUPPRESS(tag) [[gsl::suppress(tag)]]
#else
#define GSL_SUPPRESS(tag)
#endif

#if defined(__GNUC__)
#if __GNUC_PREREQ(4, 9)
#define LOTUS_EXPORT [[gnu::visibility("default")]]
#else
#define LOTUS_EXPORT __attribute__((__visibility__("default")))
#endif
#else
#define LOTUS_EXPORT
#endif

inline void MakeStringInternal(std::ostringstream& /*ss*/) noexcept {}

template <typename T>
inline void MakeStringInternal(std::ostringstream& ss, const T& t) noexcept {
  ss << t;
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::ostringstream& ss, const T& t, const Args&... args) noexcept {
  ::onnxruntime::MakeStringInternal(ss, t);
  ::onnxruntime::MakeStringInternal(ss, args...);
}

template <typename... Args>
std::string MakeString(const Args&... args) {
  std::ostringstream ss;
  ::onnxruntime::MakeStringInternal(ss, args...);
  return std::string(ss.str());
}

// Specializations for already-a-string types.
template <>
inline std::string MakeString(const std::string& str) {
  return str;
}
inline std::string MakeString(const char* p_str) {
  return p_str;
}

inline long long TimeDiffMicroSeconds(TimePoint start_time) {
  auto end_time = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
}

inline long long TimeDiffMicroSeconds(TimePoint start_time, TimePoint end_time) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
}

inline std::string GetCurrentTimeString() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm;  //NOLINT

#ifdef _WIN32
  localtime_s(&local_tm, &in_time_t);
#else
  localtime_r(&in_time_t, &local_tm);
#endif

  char time_str[32];
  strftime(time_str, sizeof(time_str), "%Y-%m-%d_%H-%M-%S", &local_tm);
  return std::string(time_str);
}

struct null_type {};

inline size_t Align256(size_t v) {
  return (v + 255) & ~static_cast<size_t>(255);
}

}  // namespace onnxruntime
