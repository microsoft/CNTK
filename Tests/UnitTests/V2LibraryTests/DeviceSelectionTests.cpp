//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;

namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(DeviceSelectionSuite)

BOOST_AUTO_TEST_CASE(SetCpuDeviceAsDefault)
{
    auto cpuDevice = DeviceDescriptor::CPUDevice();
    DeviceDescriptor::SetDefaultDevice(cpuDevice);

    BOOST_TEST((DeviceDescriptor::DefaultDevice() == cpuDevice));
}

BOOST_AUTO_TEST_CASE(SetBestDeviceAsDefault)
{
    auto bestDevice = DeviceDescriptor::BestDevice();
    DeviceDescriptor::SetDefaultDevice(bestDevice);

    BOOST_TEST((DeviceDescriptor::DefaultDevice() == bestDevice));
}

BOOST_AUTO_TEST_CASE(UseCpuDeviceAsDefault)
{
    auto cpuDevice = DeviceDescriptor::CPUDevice();
    auto bestDevice = DeviceDescriptor::BestDevice();

    if (bestDevice != cpuDevice)
    {
        DeviceDescriptor::SetDefaultDevice(cpuDevice);
    }

    BOOST_TEST((DeviceDescriptor::UseDefaultDevice() == cpuDevice));

    if (DeviceDescriptor::DefaultDevice() != bestDevice)
    {
        VerifyException([&bestDevice]() {
            DeviceDescriptor::SetDefaultDevice(bestDevice);
        }, "Was able to invoke SetDefaultDevice() after UseDefaultDevice().");
    }
}

BOOST_AUTO_TEST_CASE(CpuAndBestDevicesInAllDevices)
{
    auto cpuDevice = DeviceDescriptor::CPUDevice();
    DeviceDescriptor::SetDefaultDevice(cpuDevice);

    BOOST_TEST((DeviceDescriptor::UseDefaultDevice() == cpuDevice));

    // Invoke BestDevice after releasing the lock in UseDefaultDevice().
    auto bestDevice = DeviceDescriptor::BestDevice();

    const auto& allDevices = DeviceDescriptor::AllDevices();

#ifdef CPUONLY
    BOOST_TEST(allDevices.size() == 1);
#endif
    auto numGpuDevices = allDevices.size() - 1;

    VerifyException([&numGpuDevices]() {
        DeviceDescriptor::GPUDevice((unsigned int)numGpuDevices);
    }, "Was able to create GPU device descriptor with invalid id.");

    BOOST_TEST((find(allDevices.begin(), allDevices.end(), bestDevice) != allDevices.end()));
    BOOST_TEST((allDevices.back() == cpuDevice));
}

BOOST_AUTO_TEST_SUITE_END()

}}
