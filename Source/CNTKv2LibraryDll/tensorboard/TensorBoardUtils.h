//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"

namespace tensorflow
{
    class GraphDef;
}

namespace CNTK 
{
    namespace Internal 
    {
        ///
        /// Populates the given TensorBoard GraphDef with the graph of the given CNTK function.
        ///
        void CreateTensorBoardGraph(const FunctionPtr& src, tensorflow::GraphDef& dst);

    #ifndef CNTK_UWP
        void WriteImageToBuffer(void* matrix, DataType dtype, int height, int width, int depth, std::vector<unsigned char>& buffer);
    #endif // !CNTK_UWP

    }
}
