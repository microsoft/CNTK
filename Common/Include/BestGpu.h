//
// <copyright file="BestGPU.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

// #define CPUONLY      // #define this to build without GPU support nor needing the SDK installed
#include "CommonMatrix.h"

// define IConfigRecord and ConfigParameters as incomplete types, in order to avoid having to include "ScriptableObjects.h" and "commandArgUtil.h", as that confuses some .CU code
namespace Microsoft { namespace MSR { namespace ScriptableObjects {
    struct IConfigRecord;
}}}

namespace Microsoft { namespace MSR { namespace CNTK {
#ifndef CPUONLY
    class ConfigParameters;
    DEVICEID_TYPE DeviceFromConfig(const ConfigParameters & config);
    DEVICEID_TYPE DeviceFromConfig(const ScriptableObjects::IConfigRecord & config);
#else
    template<class ConfigRecordType>
    static inline DEVICEID_TYPE DeviceFromConfig(const ConfigRecordType & /*config*/) { return -1/*CPUDEVICE*/; }    // tells runtime system to not try to use GPUs
    // TODO: find a way to use CPUDEVICE without a huge include overhead; OK so far since CPUONLY mode is sorta special...
#endif
}}}
