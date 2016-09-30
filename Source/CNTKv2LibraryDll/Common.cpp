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
        void DisableAutomaticUnpackingOfPackedValues()
        {
            s_disableAutomaticUnpackingOfPackedValues.store(true);
        }

        bool IsAutomaticUnpackingOfPackedValuesDisabled()
        {
            return s_disableAutomaticUnpackingOfPackedValues.load();
        }
    }

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

    /*static*/ Axis::UniqueDynamicAxesNames Axis::s_uniqueDynamicAxisNames;

    /*static*/ const std::vector<Axis> Axis::DefaultInputVariableDynamicAxes = { Axis::DefaultDynamicAxis(), Axis::DefaultBatchAxis() };

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

    /*static*/ Axis Axis::NewUniqueDynamicAxis(const std::wstring& axisNamePrefix, bool isOrderedDynamicAxis /*= true*/)
    {
        return Axis(s_uniqueDynamicAxisNames.NewUniqueDynamicAxisName(axisNamePrefix), isOrderedDynamicAxis);
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
