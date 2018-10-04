// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
// NOTE: Don't include this file directly. Include logging.h

#define CREATE_MESSAGE(logger, severity, category, datatype) \
  ::onnxruntime::Logging::Capture(logger, ::onnxruntime::Logging::Severity::k##severity, category, datatype, WHERE)

/*
Both printf and stream style logging are supported.
Not that printf currently has a 2K limit to the message size.

LOGS_* macros are for stream style
LOGF_* macros are for printf style

The Message class captures the log input, and pushes it through the logger in its destructor.

Use the *FATAL* macros if you want a Severity::kFatal message to also throw.

There are a few variants to minimize the length of the macro name required in the calling code. 
They are optimized so the shortest names are for the (expected) most common usage. This can be
tweaked if needed.

Explicit logger vs LoggingManager::DefaulLogger()
  Default is for a logger instance to be explicitly passed in. 
  The logger instance provides an identifier so that log messages from different runs can be separated.

  Variants with DEFAULT in the macro name use the default logger provided by logging manager. This is
  static so accessible from any code, provided a LoggingManager instance created with InstanceType::Default 
  exists somewhere. See logging.h for further explanation of the expected setup.
  
DataType
  Default uses DataType::SYSTEM.
  
  Variants with USER in the macro name use DataType::USER. This is data that could be PII, and may need to
  be filtered from output. LoggingManager applies this filtering.

Category
  Default category is ::onnxruntime::Logging::Category::onnxruntime.

  If you wish to provide a different category, use variants with CATEGORY in the macro name

*/

// Logging with explicit category

// iostream style logging. Capture log info in Message, and push to the logger in ~Message.
#define LOGS_CATEGORY(logger, severity, category)                                                                        \
  if ((logger).OutputIsEnabled(::onnxruntime::Logging::Severity::k##severity, ::onnxruntime::Logging::DataType::SYSTEM)) \
  CREATE_MESSAGE(logger, severity, category, ::onnxruntime::Logging::DataType::SYSTEM).Stream()

#define LOGS_USER_CATEGORY(logger, severity, category)                                                                 \
  if ((logger).OutputIsEnabled(::onnxruntime::Logging::Severity::k##severity, ::onnxruntime::Logging::DataType::USER)) \
  CREATE_MESSAGE(logger, severity, category, ::onnxruntime::Logging::DataType::USER).Stream()

// printf style logging. Capture log info in Message, and push to the logger in ~Message.
#define LOGF_CATEGORY(logger, severity, category, format_str, ...)                                                       \
  if ((logger).OutputIsEnabled(::onnxruntime::Logging::Severity::k##severity, ::onnxruntime::Logging::DataType::SYSTEM)) \
  CREATE_MESSAGE(logger, severity, category, ::onnxruntime::Logging::DataType::SYSTEM).CapturePrintf(format_str, ##__VA_ARGS__)

#define LOGF_USER_CATEGORY(logger, severity, category, format_str, ...)                                                \
  if ((logger).OutputIsEnabled(::onnxruntime::Logging::Severity::k##severity, ::onnxruntime::Logging::DataType::USER)) \
  CREATE_MESSAGE(logger, severity, category, ::onnxruntime::Logging::DataType::USER).CapturePrintf(format_str, ##__VA_ARGS__)

// Logging with category of "onnxruntime"

#define LOGS(logger, severity) \
  LOGS_CATEGORY(logger, severity, ::onnxruntime::Logging::Category::onnxruntime)

#define LOGS_USER(logger, severity) \
  LOGS_USER_CATEGORY(logger, severity, ::onnxruntime::Logging::Category::onnxruntime)

// printf style logging. Capture log info in Message, and push to the logger in ~Message.
#define LOGF(logger, severity, format_str, ...) \
  LOGF_CATEGORY(logger, severity, ::onnxruntime::Logging::Category::onnxruntime, format_str, ##__VA_ARGS__)

#define LOGF_USER(logger, severity, format_str, ...) \
  LOGF_USER_CATEGORY(logger, severity, ::onnxruntime::Logging::Category::onnxruntime, format_str, ##__VA_ARGS__)

/*

Macros that use the default logger. 
A LoggingManager instance must be currently valid for the default logger to be available.

*/

// Logging with explicit category

#define LOGS_DEFAULT_CATEGORY(severity, category) \
  LOGS_CATEGORY(::onnxruntime::Logging::LoggingManager::DefaultLogger(), severity, category)

#define LOGS_USER_DEFAULT_CATEGORY(severity, category) \
  LOGS_USER_CATEGORY(::onnxruntime::Logging::LoggingManager::DefaultLogger(), severity, category)

#define LOGF_DEFAULT_CATEGORY(severity, category, format_str, ...) \
  LOGF_CATEGORY(::onnxruntime::Logging::LoggingManager::DefaultLogger(), severity, category, format_str, ##__VA_ARGS__)

#define LOGF_USER_DEFAULT_CATEGORY(severity, category, format_str, ...) \
  LOGF_USER_CATEGORY(::onnxruntime::Logging::LoggingManager::DefaultLogger(), severity, category, format_str, ##__VA_ARGS__)

// Logging with category of "onnxruntime"

#define LOGS_DEFAULT(severity) \
  LOGS_DEFAULT_CATEGORY(severity, ::onnxruntime::Logging::Category::onnxruntime)

#define LOGS_USER_DEFAULT(severity) \
  LOGS_USER_DEFAULT_CATEGORY(severity, ::onnxruntime::Logging::Category::onnxruntime)

#define LOGF_DEFAULT(severity, format_str, ...) \
  LOGF_DEFAULT_CATEGORY(severity, ::onnxruntime::Logging::Category::onnxruntime, format_str, ##__VA_ARGS__)

#define LOGF_USER_DEFAULT(severity, format_str, ...) \
  LOGF_USER_DEFAULT_CATEGORY(severity, ::onnxruntime::Logging::Category::onnxruntime, format_str, ##__VA_ARGS__)

/*

Conditional logging

*/

// Logging with explicit category

#define LOGS_CATEGORY_IF(boolean_expression, logger, severity, category) \
  if ((boolean_expression) == true) LOGS_CATEGORY(logger, severity, category)

#define LOGS_DEFAULT_CATEGORY_IF(boolean_expression, severity, category) \
  if ((boolean_expression) == true) LOGS_DEFAULT_CATEGORY(severity, category)

#define LOGS_USER_CATEGORY_IF(boolean_expression, logger, severity, category) \
  if ((boolean_expression) == true) LOGS_USER_CATEGORY(logger, severity, category)

#define LOGS_USER_DEFAULT_CATEGORY_IF(boolean_expression, severity, category) \
  if ((boolean_expression) == true) LOGS_USER_DEFAULT_CATEGORY(severity, category)

#define LOGF_CATEGORY_IF(boolean_expression, logger, severity, category, format_str, ...) \
  if ((boolean_expression) == true) LOGF_CATEGORY(logger, severity, category, format_str, ##__VA_ARGS__)

#define LOGF_DEFAULT_CATEGORY_IF(boolean_expression, severity, category, format_str, ...) \
  if ((boolean_expression) == true) LOGF_DEFAULT_CATEGORY(severity, category, format_str, ##__VA_ARGS__)

#define LOGF_USER_CATEGORY_IF(boolean_expression, logger, severity, category, format_str, ...) \
  if ((boolean_expression) == true) LOGF_USER_CATEGORY(logger, severity, category, format_str, ##__VA_ARGS__)

#define LOGF_USER_DEFAULT_CATEGORY_IF(boolean_expression, severity, category, format_str, ...) \
  if ((boolean_expression) == true) LOGF_USER_DEFAULT_CATEGORY(severity, category, format_str, ##__VA_ARGS__)

// Logging with category of "onnxruntime"

#define LOGS_IF(boolean_expression, logger, severity) \
  LOGS_CATEGORY_IF(boolean_expression, logger, severity, ::onnxruntime::Logging::Category::onnxruntime)

#define LOGS_DEFAULT_IF(boolean_expression, severity) \
  LOGS_DEFAULT_CATEGORY_IF(boolean_expression, severity, ::onnxruntime::Logging::Category::onnxruntime)

#define LOGS_USER_IF(boolean_expression, logger, severity) \
  LOGS_USER_CATEGORY_IF(boolean_expression, logger, severity, ::onnxruntime::Logging::Category::onnxruntime)

#define LOGS_USER_DEFAULT_IF(boolean_expression, severity) \
  LOGS_USER_DEFAULT_CATEGORY_IF(boolean_expression, severity, ::onnxruntime::Logging::Category::onnxruntime)

#define LOGF_IF(boolean_expression, logger, severity, format_str, ...) \
  LOGF_CATEGORY_IF(boolean_expression, logger, severity, ::onnxruntime::Logging::Category::onnxruntime, format_str, ##__VA_ARGS__)

#define LOGF_DEFAULT_IF(boolean_expression, severity, format_str, ...) \
  LOGF_DEFAULT_CATEGORY_IF(boolean_expression, severity, ::onnxruntime::Logging::Category::onnxruntime, format_str, ##__VA_ARGS__)

#define LOGF_USER_IF(boolean_expression, logger, severity, format_str, ...)                                  \
  LOGF_USER_CATEGORY_IF(boolean_expression, logger, severity, ::onnxruntime::Logging::Category::onnxruntime, \
                        format_str, ##__VA_ARGS__)

#define LOGF_USER_DEFAULT_IF(boolean_expression, severity, format_str, ...)                                  \
  LOGF_USER_DEFAULT_CATEGORY_IF(boolean_expression, severity, ::onnxruntime::Logging::Category::onnxruntime, \
                                format_str, ##__VA_ARGS__)

/*

Debug verbose logging of caller provided level.
Disabled in Release builds.
Use the _USER variants for VLOG statements involving user data that may need to be filtered.
*/
#define VLOGS(logger, level)                                                    \
  if (::onnxruntime::Logging::vlog_enabled && level <= (logger).VLOGMaxLevel()) \
  LOGS_CATEGORY(logger, VERBOSE, "VLOG" #level)

#define VLOGS_USER(logger, level)                                               \
  if (::onnxruntime::Logging::vlog_enabled && level <= (logger).VLOGMaxLevel()) \
  LOGS_USER_CATEGORY(logger, VERBOSE, "VLOG" #level)

#define VLOGF(logger, level, format_str, ...)                                   \
  if (::onnxruntime::Logging::vlog_enabled && level <= (logger).VLOGMaxLevel()) \
  LOGF_CATEGORY(logger, VERBOSE, "VLOG" #level, format_str, ##__VA_ARGS__)

#define VLOGF_USER(logger, level, format_str, ...)                              \
  if (::onnxruntime::Logging::vlog_enabled && level <= (logger).VLOGMaxLevel()) \
  LOGF_USER_CATEGORY(logger, VERBOSE, "VLOG" #level, format_str, ##__VA_ARGS__)

// Default logger variants
#define VLOGS_DEFAULT(level) \
  VLOGS(::onnxruntime::Logging::LoggingManager::DefaultLogger(), level)

#define VLOGS_USER_DEFAULT(level) \
  VLOGS_USER(::onnxruntime::Logging::LoggingManager::DefaultLogger(), level)

#define VLOGF_DEFAULT(level, format_str, ...) \
  VLOGF(::onnxruntime::Logging::LoggingManager::DefaultLogger(), level, format_str, ##__VA_ARGS__)

#define VLOGF_USER_DEFAULT(level, format_str, ...) \
  VLOGF_USER(::onnxruntime::Logging::LoggingManager::DefaultLogger(), level, format_str, ##__VA_ARGS__)
