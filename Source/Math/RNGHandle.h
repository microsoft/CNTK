//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPUMatrix.cpp : full implementation of all matrix functions on the CPU side
//

#pragma once

#include "CommonMatrix.h"
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

class MATH_API RNGHandle
{
public:
    static std::shared_ptr<RNGHandle> Create(DEVICEID_TYPE deviceId, unsigned long seed);
    
    virtual ~RNGHandle() {}

    DEVICEID_TYPE DeviceId() const
    {
        return m_deviceId;
    }

protected:
    RNGHandle(DEVICEID_TYPE deviceId)
        : m_deviceId(deviceId)
    {}

private:

    DEVICEID_TYPE m_deviceId;
};

}}}
