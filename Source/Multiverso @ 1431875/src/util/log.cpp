#include "multiverso/util/log.h"

#include <time.h>
#include <stdarg.h>
#include <string>

#include "multiverso/util/configure.h"

namespace multiverso {

MV_DEFINE_bool(logtostderr, false, "whether output log to stderr");

// Creates a Logger instance writing messages into STDOUT.
Logger::Logger(LogLevel level) {
  level_ = level;
  file_ = nullptr;
  is_kill_fatal_ = true;
}

// Creates a Logger instance writing messages into both STDOUT and log file.
Logger::Logger(std::string filename, LogLevel level) {
  level_ = level;
  file_ = nullptr;
  ResetLogFile(filename);
}

Logger::~Logger() {
  CloseLogFile();
}

int Logger::ResetLogFile(std::string filename) {
  CloseLogFile();
  if (filename.size() > 0) {  // try to open the log file if it is specified
#ifdef _MSC_VER
    fopen_s(&file_, filename.c_str(), "w");
#else
    file_ = fopen(filename.c_str(), "w");
#endif
    if (file_ == nullptr) {
      Write(LogLevel::Error, "Cannot create log file %s\n", filename.c_str());
      return -1;
    }
  }
  return 0;
}

void Logger::Write(LogLevel level, const char *format, ...) {
  va_list val;
  va_start(val, format);
  WriteImpl(level, format, &val);
  va_end(val);
}

void Logger::Debug(const char *format, ...) {
  va_list val;
  va_start(val, format);
  WriteImpl(LogLevel::Debug, format, &val);
  va_end(val);
}

void Logger::Info(const char *format, ...) {
  va_list val;
  va_start(val, format);
  WriteImpl(LogLevel::Info, format, &val);
  va_end(val);
}

void Logger::Error(const char *format, ...) {
  va_list val;
  va_start(val, format);
  WriteImpl(LogLevel::Error, format, &val);
  va_end(val);
}

void Logger::Fatal(const char *format, ...) {
  va_list val;
  va_start(val, format);
  WriteImpl(LogLevel::Fatal, format, &val);
  va_end(val);
}

inline void Logger::WriteImpl(LogLevel level,
    const char *format, va_list* val) {
  if (level >= level_) {  // omit the message with low level
    std::string level_str = GetLevelStr(level);
    std::string time_str = GetSystemTime();
    va_list val_copy;
    va_copy(val_copy, *val);
    // write to STDOUT
    if (MV_CONFIG_logtostderr) {
      fprintf(stderr, "[%s] [%s] ", level_str.c_str(), time_str.c_str());
      vfprintf(stderr, format, *val);
    } else {
      printf("[%s] [%s] ", level_str.c_str(), time_str.c_str());
      vprintf(format, *val);
      fflush(stdout);
    }
    // write to log file
    if (file_ != nullptr) {
      fprintf(file_, "[%s] [%s] ", level_str.c_str(), time_str.c_str());
      vfprintf(file_, format, val_copy);
      fflush(file_);
    }
    va_end(val_copy);

    if (is_kill_fatal_ && level == LogLevel::Fatal) {
      CloseLogFile();
      exit(1);
    }
  }
}

// Closes the log file if it it not null.
void Logger::CloseLogFile() {
  if (file_ != nullptr) {
    fclose(file_);
    file_ = nullptr;
  }
}

std::string Logger::GetSystemTime() {
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

std::string Logger::GetLevelStr(LogLevel level) {
  switch (level) {
  case LogLevel::Debug: return "DEBUG";
  case LogLevel::Info: return "INFO";
  case LogLevel::Error: return "ERROR";
  case LogLevel::Fatal: return "FATAL";
  default: return "UNKNOW";
  }
}
//-- End of Logger routine ----------------------------------------------/

Logger Log::logger_;    // global (in process) static Logger instance

int Log::ResetLogFile(std::string filename) {
  return logger_.ResetLogFile(filename);
}

void Log::ResetLogLevel(LogLevel level) {
  logger_.ResetLogLevel(level);
}

void Log::ResetKillFatal(bool is_kill_fatal) {
  logger_.ResetKillFatal(is_kill_fatal);
}

void Log::Write(LogLevel level, const char *format, ...) {
  va_list val;
  va_start(val, format);
  logger_.WriteImpl(level, format, &val);
  va_end(val);
}

void Log::Debug(const char *format, ...) {
  va_list val;
  va_start(val, format);
  logger_.WriteImpl(LogLevel::Debug, format, &val);
  va_end(val);
}

void Log::Info(const char *format, ...) {
  va_list val;
  va_start(val, format);
  logger_.WriteImpl(LogLevel::Info, format, &val);
  va_end(val);
}

void Log::Error(const char *format, ...) {
  va_list val;
  va_start(val, format);
  logger_.WriteImpl(LogLevel::Error, format, &val);
  va_end(val);
}

void Log::Fatal(const char *format, ...) {
  va_list val;
  va_start(val, format);
  logger_.WriteImpl(LogLevel::Fatal, format, &val);
  va_end(val);
}

}  // namespace multiverso
