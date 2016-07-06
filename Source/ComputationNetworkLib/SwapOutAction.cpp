//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "SwapOutAction.h"
#include <iostream>

#ifndef CPUONLY
	#include <cuda_runtime.h>
#endif


namespace Microsoft { namespace MSR { namespace CNTK {

using std::cout;
using std::endl;


void SwapOutAction::executeAction()
{
    // do we already have a pinned, that is page-locked buffer?
    if (!m_bufferCPU){ allocatePinnedBuffer(); }

    // perform the actual asynchronous copy
    CUDA_CALL(cudaMemcpyAsync(m_bufferCPU->Data(), m_bufferGPU->Data(), m_bufferGPU->BufferSize(), cudaMemcpyDefault, m_streamAsync));
}


void SwapOutAction::allocatePinnedBuffer()
{
    size_t cols = m_bufferGPU->GetNumCols();
    size_t rows = m_bufferGPU->GetNumRows();

    //cout << cols << "x" << rows << endl;

    //-1 = deviceId = CPU
    m_bufferCPU = new Matrix<float>(rows, cols, -1);
}

}}}


