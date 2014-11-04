//
// <copyright file="BestGPU.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

// #define CPUONLY      // #define this to build without GPU support nor needing the SDK installed

namespace Microsoft { namespace MSR { namespace CNTK {
    short DeviceFromConfig(const class ConfigParameters& config);
}}}
