#ifndef CORE_GRAPH_STATUS_H
#define CORE_GRAPH_STATUS_H

#include <string>

namespace ONNXIR
{

#define RETURN_IF_ERROR(expr)           \
  do {                                  \
    auto status = (expr);               \
    if ((!status.Ok())) return status;  \
  } while (0)

    class Status
    {
    public:
        Status() = delete;

        // Constructor.
        Status(bool p_ok, const std::string& p_errMsg);

        // Copy constructor.
        Status(const Status& p_other);

        // Getter of <m_ok>.
        bool Ok() const;

        // Getter of <m_errMsg>.
        const std::string& ErrorMsg() const;

        static Status OK();

    private:

        bool m_ok;
        std::string m_errMsg;
    };
}

#endif // !CORE_GRAPH_STATUS_H
