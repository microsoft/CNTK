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
#include <map>
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

namespace Lotus {

template <typename Key, typename Value>
using LotusMap = std::unordered_map<Key, Value>;

using TimePoint = std::chrono::high_resolution_clock::time_point;

// Using statements for common classes that we refer to in lotus very often.
// TODO(Task:137) Remove 'using' statements from header files
using Common::Status;
using Common::StatusCategory;
using Common::StatusCode;
using std::make_unique;
using std::set;
using std::string;
using std::unique_ptr;
using std::vector;

#ifdef _WIN32
#define UNUSED_PARAMETER(x) (x)
#else
#define UNUSED_PARAMETER(x)
#endif

// std::vector<std::string> GetStackTrace();

// __PRETTY_FUNCTION__ isn't a macro on gcc, so use a check for _MSC_VER
// so we only define it as one for MSVC
#if (_MSC_VER && !defined(__PRETTY_FUNCTION__))
#define __PRETTY_FUNCTION__ __FUNCTION__
#endif

// Capture where a message is coming from. Use __FUNCTION__ rather than the much longer __PRETTY_FUNCTION__
#define WHERE \
  Lotus::CodeLocation(__FILE__, __LINE__, __FUNCTION__)

#define WHERE_WITH_STACK \
  Lotus::CodeLocation(__FILE__, __LINE__, __PRETTY_FUNCTION__) // , Lotus::GetStackTrace())

// Throw an exception with optional message.
// NOTE: The arguments get streamed into a string via ostringstream::operator<<
// DO NOT use a printf format string, as that will not work as you expect.
#define LOTUS_THROW(...) throw Lotus::LotusException(WHERE_WITH_STACK, Lotus::MakeString(__VA_ARGS__))

// Just in order to mark things as not implemented. Do not use in final code.
#define LOTUS_NOT_IMPLEMENTED(...) throw Lotus::NotImplementedException(Lotus::MakeString(__VA_ARGS__))

// Check condition.
// NOTE: The arguments get streamed into a string via ostringstream::operator<<
// DO NOT use a printf format string, as that will not work as you expect.
#define LOTUS_ENFORCE(condition, ...) \
  if (!(condition)) throw Lotus::LotusException(WHERE_WITH_STACK, #condition, Lotus::MakeString(__VA_ARGS__))

#define LOTUS_MAKE_STATUS(category, code, ...) \
  Status(category, code, Lotus::MakeString(__VA_ARGS__))

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

inline void MakeStringInternal(std::stringstream& /*ss*/) noexcept {}

template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) noexcept {
  ss << t;
}

template <typename T, typename... Args>
inline void MakeStringInternal(std::stringstream& ss, const T& t, const Args&... args) noexcept {
  ::Lotus::MakeStringInternal(ss, t);
  ::Lotus::MakeStringInternal(ss, args...);
}

template <typename... Args>
string MakeString(const Args&... args) {
  std::stringstream ss;
  ::Lotus::MakeStringInternal(ss, args...);
  return string(ss.str());
}

// Specializations for already-a-string types.
template <>
inline string MakeString(const string& str) {
  return str;
}
inline string MakeString(const char* p_str) {
  return string(p_str);
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
  std::tm local_tm;

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

}  // namespace Lotus
