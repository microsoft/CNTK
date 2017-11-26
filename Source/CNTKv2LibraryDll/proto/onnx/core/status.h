#ifndef CORE_GRAPH_STATUS_H
#define CORE_GRAPH_STATUS_H

#include <memory>
#include <string>


namespace ONNXIR
{
    namespace Common
    {

#define RETURN_IF_ERROR(expr)             \
  do {                                    \
    auto _status = (expr);                \
    if ((!_status.Ok())) return _status;  \
  } while (0)

        enum StatusCategory
        {
            NONE = 0,
            SYSTEM = 1,
            ONNX = 2,
        };

        // Error code for ONNX.
        enum StatusCode
        {
            OK = 0,
            FAIL = 1,
            INVALID_ARGUMENT = 2,
            NO_SUCHFILE = 3,
            NO_MODEL = 4,
            ENGINE_ERROR = 5,
            RUNTIME_EXCEPTION = 6,
            INVALID_PROTOBUF = 7,
            MODEL_LOADED = 8,
            ONNX_NOT_IMPLEMENTED = 9,
        };

        class Status
        {
        public:

            Status() {}

            Status(StatusCategory p_category, int p_code, const std::string& p_msg);

            Status(StatusCategory p_category, int p_code);

            inline Status(const Status& p_other)
                : m_state((p_other.m_state == NULL) ? NULL : new State(*p_other.m_state)) {}

            bool Ok() const;

            int Code() const;

            StatusCategory Category() const;

            const std::string& ErrorMessage() const;

            std::string ToString() const;

            inline void operator=(const Status& p_other)
            {
                if (nullptr == p_other.m_state)
                {
                    m_state.reset();
                }
                else if (m_state != p_other.m_state)
                {
                    m_state.reset(new State(*p_other.m_state));
                }
            }

            inline bool operator==(const Status& p_other) const
            {
                return (this->m_state == p_other.m_state) || (ToString() == p_other.ToString());
            }

            inline bool operator!=(const Status& p_other) const
            {
                return !(*this == p_other);
            }

            static const Status& OK();

        private:

            static const std::string& EmptyString();

            struct State
            {
                StatusCategory m_category;
                int m_code;
                std::string m_msg;
            };

            // As long as Code() is OK, m_state == NULL.
            std::unique_ptr<State> m_state;
        };
    }
}

#endif // !CORE_GRAPH_STATUS_H
