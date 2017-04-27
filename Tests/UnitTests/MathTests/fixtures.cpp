//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"

#include "../../../Source/Math/CPUMatrix.h"
#include "../../../Source/Math/GPUMatrix.h"

using namespace Microsoft::MSR::CNTK;

unsigned long RandomSeedFixture::s_counter;

// We use this fixture at the beginning of each test case to (i) re-create the
// GPU RNG and (ii) get incrementing counters, which we use in the test as seed
// explicitly specified for each random operation.
RandomSeedFixture::RandomSeedFixture()
{
    GPUMatrix<float>::ResetCurandObject(42, __FUNCTION__);
    GPUMatrix<double>::ResetCurandObject(42, __FUNCTION__);

    s_counter = 0;
}

unsigned long RandomSeedFixture::IncrementCounter()
{
    return ++s_counter;
}

// If this is not done, some math test fail (for instance,
// ScaleAndAdd(a,b,c); produces different results from
// Scale(a,b); c+=b;)
// Please, note that the setting is not limited by the lifespan 
// of the fixure, but once set, it will remain in place for all 
// following tests that will be executed by the same process.
DeterministicCPUAlgorithmsFixture::DeterministicCPUAlgorithmsFixture()
{
    CPUMatrix<float /*any type will do*/>::SetCompatibleMode();
}