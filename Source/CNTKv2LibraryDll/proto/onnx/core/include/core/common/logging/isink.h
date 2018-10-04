// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/logging/logging.h"

namespace onnxruntime {
namespace Logging {
class ISink {
 public:
  ISink() = default;

  /**
  Sends the message to the sink.
  @param timestamp The timestamp.
  @param logger_id The logger identifier.
  @param message The captured message.
  */
  void Send(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) {
    SendImpl(timestamp, logger_id, message);
  }

  virtual ~ISink() = default;

 private:
  // Make Code Analysis happy by disabling all for now. Enable as needed.
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(ISink);

  virtual void SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) = 0;
};
}  // namespace Logging
}  // namespace onnxruntime
