#pragma once

#include <ostream>
#include <sstream>
#include <string>
#include <iostream>

#include "core/common/logging/capture.h"
#include "core/common/logging/isink.h"

namespace CNTK {
class CNTKClogSink : public Lotus::Logging::ISink {
public:
    CNTKClogSink()
        : stream_{&(std::clog)}, flush_{true}
    {}

    void SendImpl(const Lotus::Logging::Timestamp &timestamp, 
        const std::string &logger_id, const Lotus::Logging::Capture &message) override
    {
        UNUSED_PARAMETER(timestamp);

        std::ostringstream msg;

        msg << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id << ", "
            << message.Location().ToString() << "] " << message.Message();

        (*stream_) << msg.str() << "\n";

        if (flush_) {
            stream_->flush();
        }
    }

private:
    std::ostream *stream_;
    const bool flush_;
};
} // namespace CNTK