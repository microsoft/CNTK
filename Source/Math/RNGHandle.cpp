//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPUMatrix.cpp : full implementation of all matrix functions on the CPU side
//

#include "stdafx.h"
#include "RNGHandle.h"
#include "CPURNGHandle.h"
#include "GPURNGHandle.h"

namespace Microsoft { namespace MSR { namespace CNTK {

/*static*/ std::shared_ptr<RNGHandle> RNGHandle::Create(DEVICEID_TYPE deviceId, uint64_t seed, uint64_t offset)
{
    if (deviceId == CPUDEVICE)
    {
        return std::make_shared<CPURNGHandle>(deviceId, seed, offset);
    }
    else
    {
        return std::make_shared<GPURNGHandle>(deviceId, seed, offset);
    }
}

}}}
