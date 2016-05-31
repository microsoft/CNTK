//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"

namespace CNTK
{
    /*static*/ DeviceDescriptor DeviceDescriptor::DefaultDevice()
    {
        // TODO: Should return the global default device.
        return GPUDevice(0);
    }
}
