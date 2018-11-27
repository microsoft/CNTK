//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// RNGHandle.h: An abstraction around a random number generator
//

#pragma once

#include "CommonMatrix.h"
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

class MATH_API RNGHandle
{
public:
    static std::shared_ptr<RNGHandle> Create(DEVICEID_TYPE deviceId, uint64_t seed, uint64_t offset = 0);
    
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
