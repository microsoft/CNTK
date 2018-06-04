#include "proto/onnx/core/common/CommonSTD.h"
#include "proto/onnx/core/common/common.h"
#include "proto/onnx/core/common/status.h"
#include "proto/onnx/core/common/CommonSTD.h"

namespace ONNX
{
namespace Common
{
Status::Status(StatusCategory category, int code, const std::string& msg)
{
    // state_ will be allocated here causing the status to be treated as a failure
    LOTUS_ENFORCE(code != static_cast<int>(MLStatus::OK));

    state_ = std::make_unique<State>(category, code, msg);
}

Status::Status(StatusCategory category, int code)
    : Status(category, code, EmptyString())
{
}

bool Status::IsOK() const noexcept
{
    return (state_ == NULL);
}

StatusCategory Status::Category() const noexcept
{
    return IsOK() ? StatusCategory::NONE : state_->category;
}

int Status::Code() const noexcept
{
    return IsOK() ? static_cast<int>(StatusCode::OK) : state_->code;
}

const std::string& Status::ErrorMessage() const
{
    return IsOK() ? EmptyString() : state_->msg;
}

std::string Status::ToString() const
{
    if (state_ == nullptr)
    {
        return std::string("OK");
    }

    std::string result;

    if (StatusCategory::SYSTEM == state_->category)
    {
        result += "SystemError";
        result += " : ";
        result += std::to_string(errno);
    }
    else if (StatusCategory::ONNX == state_->category)
    {
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

const Status& Status::OK() noexcept
{
    static Status s_ok;
    return s_ok;
}

const std::string& Status::EmptyString()
{
    static std::string s_empty;
    return s_empty;
}
} // namespace Common
} // namespace ONNX
