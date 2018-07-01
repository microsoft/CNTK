#include "core/common/CommonSTD.h"
#include "core/common/logging/capture.h"
#include "core/common/logging/logging.h"
// #include "gsl/span"
#include "gsl/gsl_util"

namespace Lotus {
namespace Logging {

void Capture::CapturePrintf(msvc_printf_check const char *format, ...) {
  va_list arglist;
  va_start(arglist, format);

  ProcessPrintf(format, arglist);

  va_end(arglist);
}

// from https://github.com/KjellKod/g3log/blob/master/src/logcapture.cpp LogCapture::capturef
// License: https://github.com/KjellKod/g3log/blob/master/LICENSE
void Capture::ProcessPrintf(msvc_printf_check const char *format, va_list args) {
  static constexpr auto kTruncatedWarningText = "[...truncated...]";
  static const int kMaxMessageSize = 2048;
  char finished_message[kMaxMessageSize];

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) && !defined(__GNUC__))
  const auto finished_message_len = _countof(finished_message);
#else
  int finished_message_len = sizeof(finished_message);
#endif

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__) && !defined(__GNUC__))
  const int nbrcharacters = vsnprintf_s(finished_message, finished_message_len, _TRUNCATE, format, args);
#else
  const int nbrcharacters = vsnprintf(finished_message, finished_message_len, format, args);
#endif

  if (nbrcharacters <= 0) {
    stream_ << "\n\tERROR LOG MSG NOTIFICATION: Failure to successfully parse the message";
    stream_ << '"' << format << '"' << std::endl;
  } else if (static_cast<uint32_t>(nbrcharacters) > finished_message_len) {
    stream_ << finished_message << kTruncatedWarningText;
  } else {
    stream_ << finished_message;
  }
}

Capture::~Capture() {
  if (logger_ != nullptr) {
    logger_->Log(*this);
  }
}
}  // namespace Logging
}  // namespace Lotus
