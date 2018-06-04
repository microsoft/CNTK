#pragma once

#include <memory>
#include <string>
#include "proto/onnx/core/common/common.h"
#include "proto/onnx/core/inc/ml_status.h"

namespace ONNX
{
namespace Common
{

#define RETURN_IF_ERROR(expr)  \
    do                         \
    {                          \
        auto _status = (expr); \
        if ((!_status.IsOK())) \
            return _status;    \
    } while (0)

enum StatusCategory
{
    NONE = 0,
    SYSTEM = 1,
    ONNX = 2,
};

// Error code for lotus.
enum StatusCode
{
    OK = static_cast<unsigned int>(MLStatus::OK),
    FAIL = static_cast<unsigned int>(MLStatus::FAIL),
    INVALID_ARGUMENT = static_cast<unsigned int>(MLStatus::INVALID_ARGUMENT),
    NO_SUCHFILE = static_cast<unsigned int>(MLStatus::NO_SUCHFILE),
    NO_MODEL = static_cast<unsigned int>(MLStatus::NO_MODEL),
    ENGINE_ERROR = static_cast<unsigned int>(MLStatus::ENGINE_ERROR),
    RUNTIME_EXCEPTION = static_cast<unsigned int>(MLStatus::RUNTIME_EXCEPTION),
    INVALID_PROTOBUF = static_cast<unsigned int>(MLStatus::INVALID_PROTOBUF),
    MODEL_LOADED = static_cast<unsigned int>(MLStatus::MODEL_LOADED),
    MLStatus_NOT_IMPLEMENTED = static_cast<unsigned int>(MLStatus::MLStatus_NOT_IMPLEMENTED),
    INVALID_GRAPH = static_cast<unsigned int>(MLStatus::INVALID_GRAPH),
};

class Status
{
public:
    Status() noexcept {}

    Status(StatusCategory category, int code, const std::string& msg);

    Status(StatusCategory category, int code);

    Status(const Status& other)
        : state_((other.state_ == nullptr) ? nullptr : std::make_unique<State>(*other.state_)) {}

    void operator=(const Status& other)
    {
        if (state_ != other.state_)
        {
            if (other.state_ == nullptr)
            {
                state_.reset();
            }
            else
            {
                state_ = std::make_unique<State>(*other.state_);
            }
        }
    }

    Status(Status&& other) = default;
    Status& operator=(Status&& other) = default;
    ~Status() = default;

    bool IsOK() const noexcept;

    int Code() const noexcept;

    StatusCategory Category() const noexcept;

    const std::string& ErrorMessage() const;

    std::string ToString() const;

    bool operator==(const Status& other) const
    {
        return (this->state_ == other.state_) || (ToString() == other.ToString());
    }

    bool operator!=(const Status& other) const
    {
        return !(*this == other);
    }

    static const Status& OK() noexcept;

private:
    static const std::string& EmptyString();

    struct State
    {
        State(StatusCategory cat0, int code0, const std::string& msg0)
            : category(cat0), code(code0), msg(msg0) {}

        StatusCategory category = StatusCategory::NONE;
        int code = 0;
        std::string msg;
    };

    // As long as Code() is OK, state_ == nullptr.
    std::unique_ptr<State> state_;
};

inline std::ostream& operator<<(std::ostream& out, const Status& status)
{
    return out << status.ToString();
}

} // namespace Common
} // namespace ONNX
