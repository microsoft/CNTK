// Test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

int main()
{
    CNTK::DeviceDescriptor dd = CNTK::DeviceDescriptor::GPUDevice(0);
    bool ok = CNTK::DeviceDescriptor::TrySetDefaultDevice(dd);

    std::cout << ok << std::endl;

    return 0;
}

