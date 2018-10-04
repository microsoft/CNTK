// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/status.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace common {
Status::Status(StatusCategory category, int code, const std::string& msg) {
  // state_ will be allocated here causing the status to be treated as a failure
  LOTUS_ENFORCE(code != static_cast<int>(MLStatus::OK));

  state_ = std::make_unique<State>(category, code, msg);
}

Status::Status(StatusCategory category, int code)
    : Status(category, code, EmptyString()) {
}

bool Status::IsOK() const noexcept {
  return (state_ == nullptr);
}

StatusCategory Status::Category() const noexcept {
  return IsOK() ? common::NONE : state_->category;
}

int Status::Code() const noexcept {
  return IsOK() ? static_cast<int>(common::OK) : state_->code;
}

const std::string& Status::ErrorMessage() const noexcept {
  return IsOK() ? EmptyString() : state_->msg;
}

std::string Status::ToString() const {
  if (state_ == nullptr) {
    return std::string("OK");
  }

  std::string result;

  if (common::SYSTEM == state_->category) {
    result += "SystemError";
    result += " : ";
    result += std::to_string(errno);
  } else if (common::LOTUS == state_->category) {
    result += "[LotusError]";
    result += " : ";
    result += std::to_string(Code());
    std::string msg;

    result += " : ";
    result += MLStatusToString(static_cast<MLStatus>(Code()));
    result += " : ";
    result += state_->msg;
  }

  return result;
}

// GSL_SUPRESS(i.22) is broken. Ignore the warnings for the static local variables that are trivial
// and should not have any destruction order issues via pragmas instead.
// https://developercommunity.visualstudio.com/content/problem/249706/gslsuppress-does-not-work-for-i22-c-core-guideline.html
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26426)
#endif
const Status& Status::OK() noexcept {
  static Status s_ok;
  return s_ok;
}

const std::string& Status::EmptyString() noexcept {
  static std::string s_empty;
  return s_empty;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

}  // namespace common
}  // namespace onnxruntime
