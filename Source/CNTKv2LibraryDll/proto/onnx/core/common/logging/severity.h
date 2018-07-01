#pragma once

namespace Lotus {
namespace Logging {
// mild violation of naming convention. the 'k' lets us use token concatenation in the macro
// Lotus::Logging::Severity::k##severity. It's not legal to have Lotus::Logging::Severity::##severity
// the uppercase makes the LOG macro usage look as expected for passing an enum value as it will be LOGS(logger, ERROR)
enum class Severity {
  kVERBOSE = 0,
  kINFO = 1,
  kWARNING = 2,
  kERROR = 3,
  kFATAL = 4
};

const char SEVERITY_PREFIX[] = "VIWEF";

}  // namespace Logging
}  // namespace Lotus
