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

    /*static*/ const Axis& Axis::DefaultDynamicAxis()
    {
        static Axis s_defaultDynamicAxis(L"defaultDynamicAxis");
        return s_defaultDynamicAxis;
    }

    /*static*/ const Axis& Axis::BatchAxis()
    {
        static Axis s_batchAxis(L"batchAxis");
        return s_batchAxis;
    }

    /*static*/ const Axis& Axis::AllAxes()
    {
        static Axis s_allAxes(L"allAxes");
        return s_allAxes;
    }
}
