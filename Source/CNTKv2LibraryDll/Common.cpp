//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"

namespace CNTK
{
    /*static*/ std::atomic<bool> DeviceDescriptor::s_defaultDeviceFrozen(false);
    /*static*/ std::shared_ptr<DeviceDescriptor> DeviceDescriptor::s_defaultDevice(new DeviceDescriptor(DeviceDescriptor::GPUDevice(0)));

    /*static*/ DeviceDescriptor DeviceDescriptor::DefaultDevice()
    {
        return *s_defaultDevice;
    }

    /*static*/ DeviceDescriptor DeviceDescriptor::UseDefaultDevice()
    {
        s_defaultDeviceFrozen.store(true);
        return DefaultDevice();
    }

    /*static*/ void DeviceDescriptor::SetDefaultDevice(const DeviceDescriptor& newDefaultDevice)
    {
        if (s_defaultDeviceFrozen.load())
            RuntimeError("Process wide default device cannot be changed since it has been frozen by being implicitly used as the default device in a CNTK API call");

        s_defaultDevice.reset(new DeviceDescriptor(newDefaultDevice));
    }

    /*static*/ const std::wstring Axis::StaticAxisNamePrefix = L"staticAxis_";

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
}
