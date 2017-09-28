//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;

namespace CNTK { namespace Test {

struct DeviceSelectionTestFixture
{
    DeviceSelectionTestFixture()
    {
        Reset();
    }

    bool SkipLockCheck()
    {
#ifdef WIN32
        return false;
#else
        return true;
#endif
    }


    void Reset() {
        DeviceDescriptor::Reset();
    }

    // this returns true, if there's a lock on the given device (only works on Windows)
    bool IsLocked(const DeviceDescriptor& device) 
    {
        return std::async([&device] { return device.IsLocked(); }).get();
    }

    std::vector<DeviceDescriptor> GetGpuDevices() 
    {
#ifdef CPUONLY
        return {};
#endif
        std::vector<DeviceDescriptor> gpuDevices;
        const auto& devices = DeviceDescriptor::AllDevices();
        std::copy_if(devices.begin(), devices.end(), back_inserter(gpuDevices),
            [](const DeviceDescriptor& device) { return device.Type() == DeviceKind::GPU; });
        return gpuDevices;
    }
    
};

BOOST_FIXTURE_TEST_SUITE(DeviceSelectionSuite, DeviceSelectionTestFixture)

BOOST_AUTO_TEST_CASE(TestAllDevices)
{
    const auto& devices = DeviceDescriptor::AllDevices();

    BOOST_TEST(any_of(devices.begin(), devices.end(),
        [](const DeviceDescriptor& device) { return device.Type() == DeviceKind::CPU; }));

#ifdef CPUONLY
        BOOST_TEST((devices.size() == 1));
#endif

    BOOST_TEST((devices.size() >= 1));

    if (devices.size() > 1) 
    {
        auto gpuDevices = GetGpuDevices();
        BOOST_TEST(gpuDevices.size() > 0);
    }
    
}

BOOST_AUTO_TEST_CASE(TestInvalidGPUDevice)
{
    const auto& devices = DeviceDescriptor::AllDevices();
    auto numGpuDevices = devices.size() - 1;
    
    VerifyException([&numGpuDevices]() {
        DeviceDescriptor::GPUDevice((unsigned int)numGpuDevices);
    }, "Was able to create GPU device descriptor with invalid id.");
}

BOOST_AUTO_TEST_CASE(TestGPUProperties)
{
    VerifyException([] {
        DeviceDescriptor::GetGPUProperties(DeviceDescriptor::CPUDevice());
    }, "Was able to retrieve GPU properties for a CPU device.");

    auto gpuDevices = GetGpuDevices();

    for (const auto& device : gpuDevices)
    {
        const auto& properties = DeviceDescriptor::GetGPUProperties(device);
        BOOST_TEST((properties.deviceId == device.Id()));
        BOOST_TEST((properties.cudaCores >= 0));
        BOOST_TEST((properties.totalMemory >= 0));
        BOOST_TEST((properties.versionMajor >= 0));
    }
}

BOOST_AUTO_TEST_CASE(TestSetCpuDeviceAsDefault)
{
    auto cpuDevice = DeviceDescriptor::CPUDevice();
    BOOST_TEST(!IsLocked(cpuDevice));
    
    // CPU device cannot be locked.
    BOOST_TEST(!DeviceDescriptor::TrySetDefaultDevice(cpuDevice, true));
    BOOST_TEST(!IsLocked(cpuDevice));
    
    BOOST_TEST(DeviceDescriptor::TrySetDefaultDevice(cpuDevice, false));
    BOOST_TEST(!IsLocked(cpuDevice));
    BOOST_TEST((DeviceDescriptor::UseDefaultDevice() == cpuDevice));
}

BOOST_AUTO_TEST_CASE(TestSetGpuDeviceAsDefault)
{
    auto gpuDevices = GetGpuDevices();
    if (gpuDevices.size() == 0)
        return;

    for (const auto& device : gpuDevices) 
    {
        Reset();
        BOOST_TEST(DeviceDescriptor::TrySetDefaultDevice(device, false));
        BOOST_TEST((DeviceDescriptor::UseDefaultDevice() == device));
        if (!device.IsLocked()) 
        {
            BOOST_TEST((SkipLockCheck() || !IsLocked(device)));
            BOOST_TEST(DeviceDescriptor::TrySetDefaultDevice(device, true));
            BOOST_TEST((DeviceDescriptor::UseDefaultDevice() == device));
            BOOST_TEST((SkipLockCheck() || IsLocked(device)));
        }
    }

    VerifyException([]() 
    { 
        DeviceDescriptor::TrySetDefaultDevice(DeviceDescriptor::CPUDevice());
    }, "Was able to invoke SetDefaultDevice() after UseDefaultDevice().");
}

BOOST_AUTO_TEST_CASE(TestSuccessiveSetDefaultDevice)
{
    auto gpuDevices = GetGpuDevices();
    if (gpuDevices.size() == 0)
        return;

    auto& device = gpuDevices[0];

    BOOST_TEST(DeviceDescriptor::TrySetDefaultDevice(device, true));
    BOOST_TEST((SkipLockCheck() || IsLocked(device)));

    BOOST_TEST(DeviceDescriptor::TrySetDefaultDevice(device, true));
    BOOST_TEST((SkipLockCheck() || IsLocked(device)));

    BOOST_TEST((DeviceDescriptor::UseDefaultDevice() == device));

    BOOST_TEST(DeviceDescriptor::TrySetDefaultDevice(device, true));
    BOOST_TEST((SkipLockCheck() || IsLocked(device)));

    Reset();

    BOOST_TEST((SkipLockCheck() || IsLocked(device)));
    BOOST_TEST(DeviceDescriptor::TrySetDefaultDevice(device, true));
    BOOST_TEST((SkipLockCheck() || IsLocked(device)));


    Reset();

    BOOST_TEST((SkipLockCheck() || IsLocked(device)));
    
    BOOST_TEST(!DeviceDescriptor::TrySetDefaultDevice(DeviceDescriptor::CPUDevice(), true));
    BOOST_TEST((SkipLockCheck() || IsLocked(device)));

    BOOST_TEST(DeviceDescriptor::TrySetDefaultDevice(DeviceDescriptor::CPUDevice(), false));
    BOOST_TEST((SkipLockCheck() || !IsLocked(device)));
}

BOOST_AUTO_TEST_CASE(TestDefaultDeviceSelection)
{
    const auto& devices = DeviceDescriptor::AllDevices();
    for (const auto& device : devices) 
    {
        BOOST_TEST((SkipLockCheck() || !IsLocked(device)));
    }

    const auto& defaultDevice = DeviceDescriptor::UseDefaultDevice();
    if (defaultDevice.Type() != DeviceKind::CPU)
    {
        BOOST_TEST((SkipLockCheck() || IsLocked(defaultDevice)));
    }

    auto gpuDevices = GetGpuDevices();
    if (gpuDevices.size() == 0)
        return;


    if (any_of(gpuDevices.begin(), gpuDevices.end(),
        [](const DeviceDescriptor& device) { return !device.IsLocked(); })) 
    {
        // if there's at least one gpu device, not locked by another process, then
        // the default device must be a gpu device.
        BOOST_TEST((defaultDevice.Type() == DeviceKind::GPU));
    }
}

BOOST_AUTO_TEST_CASE(TestDefaultDeviceSelectionWithExcludedDevices)
{
    const auto& allDevices = DeviceDescriptor::AllDevices();
    DeviceDescriptor::SetExcludedDevices(allDevices);

    VerifyException([]()
    {
        DeviceDescriptor::UseDefaultDevice();
    }, "UseDefaultDevice() didn't throw an exception with all physical devices excluded.");

    BOOST_TEST(!DeviceDescriptor::TrySetDefaultDevice(DeviceDescriptor::CPUDevice()));

    Reset();

    auto gpuDevices = GetGpuDevices();

    if (gpuDevices.size() == 0)
        return;

    DeviceDescriptor::SetExcludedDevices(gpuDevices);
    auto defaultDevice = DeviceDescriptor::UseDefaultDevice();
    BOOST_TEST((defaultDevice.Type() == DeviceKind::CPU));

    if (allDevices.size() == 1)
        return;

    Reset();
    defaultDevice = DeviceDescriptor::UseDefaultDevice();

    Reset();
    DeviceDescriptor::SetExcludedDevices({ defaultDevice });
    auto newDefaultDevice = DeviceDescriptor::UseDefaultDevice();
    BOOST_TEST((defaultDevice != newDefaultDevice));
}



BOOST_AUTO_TEST_SUITE_END()

}}
