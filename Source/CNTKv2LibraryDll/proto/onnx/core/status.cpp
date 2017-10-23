#include "status.h"

namespace ONNXIR
{
    Status::Status(bool p_ok, const std::string& p_errMsg)
    {
        m_ok = p_ok;
        m_errMsg = p_errMsg;
    }

    Status::Status(const Status& p_other)
    {
        m_ok = p_other.m_ok;
        m_errMsg = p_other.m_errMsg;
    }

    bool Status::Ok() const
    {
        return m_ok;
    }

    const std::string& Status::ErrorMsg() const
    {
        return m_errMsg;
    }

    Status Status::OK()
    {
        static Status ok(true, "");
        return ok;
    }
}
