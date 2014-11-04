//
// <copyright file="BestGPU.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#ifndef CPUONLY
#pragma comment (lib, "cudart.lib")
#include <cuda_runtime.h>
#include <nvml.h>
#include <vector>
#endif
#include "commandArgUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    short DeviceFromConfig(const ConfigParameters& config);
}}}
