#ifndef LOGREG_UTIL_LOG_H_
#define LOGREG_UTIL_LOG_H_

#include <string>

namespace logreg {
enum LogLevel : int {
  Debug = 0,
  Info = 1,
  Error = 2,
  Fatal = 3
};
class Log {
public:
  // print log to stdout
  static void Write(LogLevel level, const char *format, ...);
  static LogLevel& log_level() { return log_level_; }
private:
  static LogLevel log_level_;
  static std::string GetSystemTime();
};

#define LR_CHECK(condition)           \
if (!(condition)) {                   \
  Log::Write(Fatal, "Check failed: "  \
    #condition " at %s, line %d .\n", \
    __FILE__, __LINE__);              \
}

#ifdef LOGLEVEL_DEBUG
#define DEBUG_CHECK(condition)      \
  LR_CHECK(condition)                
#else
#define DEBUG_CHECK(condition) 
#endif


}  // namespace logreg

#endif  // LOGREG_UTIL_LOG_H_
