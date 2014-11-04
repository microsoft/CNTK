//
// <copyright file="MatrixUnitTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "CppUnitTest.h"
#include "..\Math\GPUWatcher.h"

#define epsilon 0.000001
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) // 0 based indexing

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#if 0   // BUGBUG: This somehow does not link, GetFreeMemoryOnCUDADevice is missing from CNTKMath.dll. Commening out so that we can at least get all other tests to run.
namespace CNTKMathTest
{    
    TEST_CLASS(GPUWatcherTests)
    {        

    public:

        //This test should fail if you don't have CUDA GPU (or working under remote desktop)
        TEST_METHOD(GetFreeMemoryOnCUDADeviceTest)
        {
            size_t x = GPUWatcher::GetFreeMemoryOnCUDADevice(0);
            Assert::IsTrue(x>0);
        }
    };
}
#endif
