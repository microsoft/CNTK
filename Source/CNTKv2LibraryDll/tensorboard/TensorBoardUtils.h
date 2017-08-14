//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core/types.hpp"

namespace tensorflow
{
    class GraphDef;
}

namespace CNTK
{
namespace Internal {
///
/// Populates the given TensorBoard GraphDef with the graph of the given CNTK function.
///
void CreateTensorBoardGraph(const FunctionPtr& src, tensorflow::GraphDef& dst);

void writeImageToBuffer(void* matrix, int h, int w, int type, std::vector<uchar>& buf);

}
 
}