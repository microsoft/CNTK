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

#ifndef CPUONLY
struct ProcessorData
{
	int cores;
	nvmlMemory_t memory;
	nvmlUtilization_t utilization;
	cudaDeviceProp deviceProp;
	size_t cudaFreeMem;
	size_t cudaTotalMem;
	bool dbnFound;
	bool cnFound;
	int deviceId; // the deviceId (cuda side) for this processor
};

enum BestGpuFlags
{
    bestGpuNormal = 0,
    bestGpuAvoidSharing = 1, // don't share with other known machine learning Apps (cl.exe/dbn.exe)
    bestGpuFavorMemory = 2, // favor memory
    bestGpuFavorUtilization = 4, // favor low utilization
    bestGpuFavorSpeed = 8, // favor fastest processor
    bestGpuRequery = 256, // rerun the last query, updating statistics
};

class BestGpu
{
private:
    bool m_initialized; // initialized
    bool m_nvmlData; // nvml Data is valid
    bool m_cudaData; // cuda Data is valid
    int m_deviceCount; // how many devices are available?
    int m_queryCount; // how many times have we queried the usage counters?
    BestGpuFlags m_lastFlags; // flag state at last query
    int m_lastCount; // count of devices (with filtering of allowed Devices)
    std::vector<ProcessorData*> m_procData;
    int m_allowedDevices; // bitfield of allowed devices
    void GetCudaProperties();
    void GetNvmlData();
    void QueryNvmlData();

public:
    BestGpu() : m_initialized(false), m_nvmlData(false), m_cudaData(false), m_deviceCount(0), m_queryCount(0), 
                m_lastFlags(bestGpuNormal), m_lastCount(0), m_allowedDevices(-1) 
    { Init(); }
    ~BestGpu();
    void Init();
    void SetAllowedDevices(const std::vector<int>& devices); // only allow certain GPUs
    bool DeviceAllowed(int device);
    void AllowAll(); // reset to allow all GPUs (no allowed list)
    bool UseMultiple(); // using multiple GPUs?
    int GetDevice(BestGpuFlags flags = bestGpuNormal); // get a single device
    static const int AllDevices = -1;  // can be used to specify all GPUs in GetDevices() call
    static const int RequeryDevices = -2;  // Requery refreshing statistics and picking the same number as last query
    std::vector<int> GetDevices(int number = AllDevices, BestGpuFlags flags = bestGpuNormal); // get multiple devices
};
extern BestGpu* g_bestGpu;

#endif

}}}
