//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

// #define CPUONLY      // #define this to build without GPU support nor needing the SDK installed
#include "CommonMatrix.h"

#include <vector>

// define IConfigRecord and ConfigParameters as incomplete types, in order to avoid having to include "ScriptableObjects.h" and "Config.h", as that confuses some .CU code
namespace Microsoft { namespace MSR { namespace ScriptableObjects { struct IConfigRecord; }}}

namespace Microsoft { namespace MSR { namespace CNTK {

#ifndef CPUONLY
enum class GpuValidity
{
    Valid,
    UnknownDevice,
    ComputeCapabilityNotSupported
};

struct GpuData
{
    int versionMajor;
    int versionMinor;
    int deviceId;
    int cudaCores;
    GpuValidity validity;
    string name;
    size_t totalMemory;
    size_t freeMemory;
    GpuData(int versionMajor, int versionMinor, int deviceId, int cudaCores, GpuValidity validity, const string& name, size_t totalMemory, size_t freeMemory)
        :versionMajor(versionMajor), versionMinor(versionMinor), deviceId(deviceId), cudaCores(cudaCores), validity(validity), name(name), totalMemory(totalMemory), freeMemory(freeMemory)
    {
    }

};

std::vector<GpuData> GetAllGpusData();
GpuData GetGpuData(DEVICEID_TYPE deviceId);

class ConfigParameters;
DEVICEID_TYPE DeviceFromConfig(const ConfigParameters& config);
DEVICEID_TYPE DeviceFromConfig(const ScriptableObjects::IConfigRecord& config);

// Returns an id of the best available (not exclusively locked) GPU device,
// if no GPU is available, defaults to the CPU device. Additionally, it acquires
// a lock on the selected GPU device.
DEVICEID_TYPE GetBestDevice(const vector<int>& excluded);

void ReleaseLock();
bool TryLock(int deviceId);
bool IsLocked(int deviceId);

#else

static inline DEVICEID_TYPE GetBestDevice(const vector<int>& excluded)
{
    if (std::find(excluded.begin(), excluded.end(), CPUDEVICE) != excluded.end())
        RuntimeError("Device selection: No eligible device found.");

    return CPUDEVICE;
}

static inline bool TryLock(int /*deviceId*/) 
{
    return false;
}

static inline bool IsLocked(int /*deviceId*/)
{
    return false;
}

static inline void ReleaseLock() {}

template <class ConfigRecordType>
static inline DEVICEID_TYPE DeviceFromConfig(const ConfigRecordType& /*config*/)
{
    return CPUDEVICE;
} // tells runtime system to not try to use GPUs
// TODO: find a way to use CPUDEVICE without a huge include overhead; OK so far since CPUONLY mode is sorta special...
#endif

} } }
