//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "BestGpu.h"
#include <mutex>
#include <algorithm>
#include <CPUMatrix.h> // For CPUMatrix::SetNumThreads
#include <thread>
#include "GPUMatrix.h"
#include "Globals.h"

namespace CNTK
{
    namespace Internal
    {
        static std::atomic<unsigned long long> s_nextUniqueId(0);
        size_t NewUniqueId()
        {
            return s_nextUniqueId++;
        }

        std::atomic<bool> s_reverseTensorShapesInErrorMessages(false);
        void EnableReversingTensorShapesInErrorMessages()
        {
            s_reverseTensorShapesInErrorMessages.store(true);
        }

        bool IsReversingTensorShapesInErrorMessagesEnabled()
        {
            return s_reverseTensorShapesInErrorMessages.load();
        }

        std::atomic<bool> s_alwaysAllowSettingDefaultDevice(false);
        void AlwaysAllowSettingDefaultDevice()
        {
            s_alwaysAllowSettingDefaultDevice.store(true);
        }

        bool IsSettingDefaultDeviceAlwaysAllowed()
        {
            return s_alwaysAllowSettingDefaultDevice.load();
        }

        std::atomic<bool> s_disableAutomaticUnpackingOfPackedValues(false);
        void SetAutomaticUnpackingOfPackedValues(bool disable)
        {
            s_disableAutomaticUnpackingOfPackedValues.store(disable);
        }

        bool IsAutomaticUnpackingOfPackedValuesDisabled()
        {
            return s_disableAutomaticUnpackingOfPackedValues.load();
        }

        std::atomic<int> s_computationNetworkTraceLevel(0);
        void SetComputationNetworkTraceLevel(int traceLevel)
        {
            s_computationNetworkTraceLevel.store(traceLevel);
        }

        int GetComputationNetworkTraceLevel()
        {
            return s_computationNetworkTraceLevel.load();
        }

        void SetGPUMemoryAllocationTraceLevel(int traceLevel)
        {
            Microsoft::MSR::CNTK::TracingGPUMemoryAllocator::SetTraceLevel(traceLevel);
        }

        void ForceSynchronousCUDAKernelExecutions()
        {
            Microsoft::MSR::CNTK::SyncGuard::EnableSync();
        }

        void ForceDeterministicAlgorithms()
        {
            Microsoft::MSR::CNTK::Globals::ForceDeterministicAlgorithms();
        }
    }

    /*static*/ const NDShape NDShape::Unknown(1, SentinelDimValueForUnknownShape);

    /*static*/ std::atomic<bool> DeviceDescriptor::s_defaultDeviceFrozen(false);
    /*static*/ std::shared_ptr<DeviceDescriptor> DeviceDescriptor::s_defaultDevice;
    /*static*/ std::shared_ptr<std::vector<DeviceDescriptor>> DeviceDescriptor::s_allDevices;

    static std::once_flag s_initDefaultDeviceFlag, s_initAllDevicesFlag;

    /*static*/ DeviceDescriptor DeviceDescriptor::DefaultDevice()
    {
        std::call_once(s_initDefaultDeviceFlag, [=]{
            s_defaultDevice.reset(new DeviceDescriptor(DeviceDescriptor::BestDevice()));
        });
        return *s_defaultDevice;
    }

    /*static*/ DeviceDescriptor DeviceDescriptor::UseDefaultDevice()
    {
        bool alreadyFrozen = s_defaultDeviceFrozen.exchange(true);
        auto selectedDevice = DefaultDevice();
        if (!alreadyFrozen)
        {
            Microsoft::MSR::CNTK::OnDeviceSelected(AsCNTKImplDeviceId(selectedDevice));
        }
        return selectedDevice;
    }

    /*static*/ void DeviceDescriptor::SetDefaultDevice(const DeviceDescriptor& newDefaultDevice)
    {
        if (newDefaultDevice == DefaultDevice())
            return;

        // As a testing backdoor we allow changing the default device even after being "used/frozen"
        if (!Internal::IsSettingDefaultDeviceAlwaysAllowed() && s_defaultDeviceFrozen.load())
            RuntimeError("Process wide default device cannot be changed since it has been frozen by being implicitly used as the default device in a CNTK API call");

        std::call_once(s_initDefaultDeviceFlag, []{
            // do nothing. This will set the flag above, in case when DefaultDevice() was never called before.
        });

        s_defaultDevice.reset(new DeviceDescriptor(newDefaultDevice));
    }
    
    /*static*/ DeviceDescriptor DeviceDescriptor::BestDevice()
    {
        //TODO: BestDevice remains locked if UseDefaultDevice is never executed
        // or if BestDevice() is invoked after UseDefaultDevice(). 
        // Should we do anything about it?
        auto id = Microsoft::MSR::CNTK::GetBestDevice();
        return id >= 0 ? DeviceDescriptor::GPUDevice(id) : DeviceDescriptor::CPUDevice();
    }

    /*static*/ const std::vector<DeviceDescriptor>& DeviceDescriptor::AllDevices()
    {
        using namespace Microsoft::MSR::CNTK;

        std::call_once(s_initAllDevicesFlag, [=]{
           s_allDevices.reset(new std::vector<DeviceDescriptor>());
#ifndef CPUONLY
           auto allGpusData = GetAllGpusData();

            for (const auto& gpuData : allGpusData)
            {
                if (gpuData.validity == GpuValidity::Valid)
                {
                    s_allDevices->push_back(DeviceDescriptor((unsigned int) gpuData.deviceId, DeviceKind::GPU));
                }
            }
#endif
            s_allDevices->push_back(DeviceDescriptor::CPUDevice());
        });

        return *s_allDevices;
    }

    /*static*/ DeviceDescriptor DeviceDescriptor::GPUDevice(unsigned int deviceId) 
    {       
        const auto& allDevices = AllDevices();
       
        if (std::none_of(allDevices.begin(), allDevices.end(), 
            [deviceId](const DeviceDescriptor& device){ return device.Type() == DeviceKind::GPU && device.Id() == deviceId; }))
        {
            InvalidArgument("Specified GPU device id (%u) is invalid.", deviceId);
        }
        return { deviceId, DeviceKind::GPU };
    }

    /*static*/ const std::wstring Axis::StaticAxisNamePrefix = L"staticAxis_";

    /*static*/ const int Axis::SentinelStaticAxisIndexValueForDynamicAxes = std::numeric_limits<int>::max();
    /*static*/ const int Axis::SentinelStaticAxisIndexValueForAllStaticAxes = std::numeric_limits<int>::max() - 1;
    /*static*/ const int Axis::SentinelStaticAxisIndexValueForUnknownAxes = std::numeric_limits<int>::max() - 2;

    /*static*/ Axis::UniqueDynamicAxesNames Axis::s_uniqueDynamicAxisNames;

    /*static*/ const std::vector<Axis> Axis::DefaultInputVariableDynamicAxes = { Axis::DefaultDynamicAxis(), Axis::DefaultBatchAxis() };
    /*static*/ const std::vector<Axis> Axis::UnknownDynamicAxes = { Axis(SentinelStaticAxisIndexValueForUnknownAxes) };

    /*static*/ const Axis& Axis::DefaultDynamicAxis()
    {
        static const Axis s_defaultDynamicAxis(L"defaultDynamicAxis");
        return s_defaultDynamicAxis;
    }

    /*static*/ const Axis& Axis::DefaultBatchAxis()
    {
        static const Axis s_defaultBatchAxis(L"defaultBatchAxis", false);
        return s_defaultBatchAxis;
    }

    /*static*/ const Axis& Axis::AllStaticAxes()
    {
        static const Axis s_allStaticAxes(SentinelStaticAxisIndexValueForAllStaticAxes);
        return s_allStaticAxes;
    }


    void Axis::RegisterAxisName(const std::wstring& axisName)
    {
        s_uniqueDynamicAxisNames.RegisterAxisName(axisName);
    }

    std::atomic<size_t> s_maxNumCPUThreads(std::thread::hardware_concurrency());
    void SetMaxNumCPUThreads(size_t numCPUThreads)
    {
        s_maxNumCPUThreads.store(numCPUThreads);
        Microsoft::MSR::CNTK::CPUMatrix<float>::SetNumThreads((int)numCPUThreads);
    }

    size_t GetMaxNumCPUThreads()
    {
        return s_maxNumCPUThreads.load();
    }
}
