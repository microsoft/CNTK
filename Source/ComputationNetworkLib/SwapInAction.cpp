//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "SwapInAction.h"

#ifndef CPUONLY
    #include <cuda.h>
#endif



namespace Microsoft { namespace MSR { namespace CNTK {


void SwapInAction::executeAction()
{
    if(!m_isSwappingToGPU)
        SwapToGPU();
    else
        SynchronizeBufferBeforeUse();
}

void SwapInAction::SwapToGPU()
{
        CUDA_CALL(cudaStreamSynchronize(m_swapOutStream));
        CUDA_CALL(cudaMemcpyAsync(m_bufferGPU->Data(), m_bufferCPU->Data(), m_bufferGPU->BufferSize(), cudaMemcpyDefault, m_swapInStream));
        m_isSwappingToGPU = true;
}

void SwapInAction::SynchronizeBufferBeforeUse()
{
    CUDA_CALL(cudaStreamSynchronize(m_swapInStream));
    m_isSwappingToGPU = false;
}

}}}
