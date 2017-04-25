#include "util/log.h"

#include <time.h>
#include <stdio.h>
#include <cstdarg>

namespace logreg {

// default in Info level
#ifdef LOGLEVEL_FATAL
  LogLevel Log::log_level_ = LogLevel::Fatal;
#elif LOGLEVEL_ERROR
  LogLevel Log::log_level_ = LogLevel::Error;
#elif LOGLEVEL_DEBUG
  LogLevel Log::log_level_ = LogLevel::Debug;
#else 
  LogLevel Log::log_level_ = LogLevel::Info;
#endif

void Log::Write(LogLevel level, const char *format, ...) {
  if (static_cast<int>(log_level_) > static_cast<int>(level)) {
    return;
  }
  std::string level_str;

  switch (level) {
  case Debug:
    level_str = "DEBUG";
    break;
  case Info:
    level_str = "INFO";
    break;
  case Error:
    level_str = "ERROR";
    break;
  case Fatal:
      level_str = "FATAL";
      break; 
  default:
    break;
  }
  va_list val;
  va_start(val, format);
  printf("[%s] [%s] ", level_str.c_str(), GetSystemTime().c_str());
  vprintf(format, val);
  fflush(stdout);
  va_end(val);

  if (level == Fatal) {
    exit(1);
  }
}

inline std::string Log::GetSystemTime() {
  time_t t = time(0);
  char str[64];
#ifdef _MSC_VER
  tm time;
  localtime_s(&time, &t);
  strftime(str, sizeof(str), "%Y-%m-%d %H:%M:%S", &time);
#else
  strftime(str, sizeof(str), "%Y-%m-%d %H:%M:%S", localtime(&t));
#endif
  return str;
}

}  // namespace logreg
