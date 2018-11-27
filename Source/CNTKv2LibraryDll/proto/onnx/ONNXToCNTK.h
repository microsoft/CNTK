//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"

namespace onnxruntime
{
    class Graph;
}

namespace CNTK
{
    namespace ONNX 
    {
        enum class ConvAutoPadType 
        {
            VALID = 0,
            SAME_UPPER = 1,
            SAME_LOWER = 2,
        };
    }

    class ONNXToCNTK
    {
    public:
        //
        // Create a CNTK graph (Function) given an ONNX graph. The function is created to use the 
        // specified computing device.
        //
        static FunctionPtr CreateGraph(onnxruntime::Graph* src, const DeviceDescriptor& computeDevice);
    };
}