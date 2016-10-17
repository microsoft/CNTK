#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;

void DeviceSelectionTests()
{
    fprintf(stderr, "\nTest device selection API..\n");

    auto cpuDevice = DeviceDescriptor::CPUDevice();
    DeviceDescriptor::SetDefaultDevice(cpuDevice);

    assert(DeviceDescriptor::DefaultDevice() == cpuDevice);

    auto bestDevice = DeviceDescriptor::BestDevice();
    DeviceDescriptor::SetDefaultDevice(bestDevice);

    assert(DeviceDescriptor::DefaultDevice() == bestDevice);

    if (bestDevice != cpuDevice)
    {
        DeviceDescriptor::SetDefaultDevice(cpuDevice);
    }
    
    assert(DeviceDescriptor::UseDefaultDevice() == cpuDevice);

    if (DeviceDescriptor::DefaultDevice() != bestDevice)
    {
        VerifyException([&bestDevice]() {
            DeviceDescriptor::SetDefaultDevice(bestDevice);
        }, "Was able to invoke SetDefaultDevice() after UseDefaultDevice().");
    }

    // Invoke BestDevice after releasing the lock in UseDefaultDevice().
    bestDevice = DeviceDescriptor::BestDevice();

    const auto& allDevices = DeviceDescriptor::AllDevices();

#ifdef CPUONLY
    assert(allDevices.size() == 1);
#endif
    auto numGpuDevices = allDevices.size() - 1;

    VerifyException([&numGpuDevices]() {
        DeviceDescriptor::GPUDevice((unsigned int)numGpuDevices);
    }, "Was able to create GPU device descriptor with invalid id.");

    assert(find(allDevices.begin(), allDevices.end(), bestDevice) != allDevices.end());
    assert(allDevices.back() == cpuDevice);

}