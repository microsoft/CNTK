//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"

namespace ONNXIR
{
    class Graph;
}

namespace CNTK
{
    class ONNXToCNTK
    {
    public:
        //
        // Create a CNTK graph (Function) given an ONNX graph. The function is created to use the 
        // specified computing device.
        //
        static FunctionPtr CreateGraph(ONNXIR::Graph* src, const DeviceDescriptor& computeDevice);
    };
}