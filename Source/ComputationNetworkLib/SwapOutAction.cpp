//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "SwapOutAction.h"
#include "GPUMatrix.h"

#ifndef CPUONLY
	#include "cuda_runtime_api.h"
#endif


namespace Microsoft { namespace MSR { namespace CNTK {


void SwapOutAction::executeAction()
{

    if (!m_bufferCPU){ allocatePinnedBuffer(); }
    
}


void SwapOutAction::allocatePinnedBuffer()
{
    size_t cols = m_bufferGPU->GetNumCols();
    size_t rows = m_bufferGPU->GetNumRows();

    float *pinnedBuffer;

    

    //CUDA_CALL(cudaHostAlloc(&pinnedBuffer, sizeof(float)*cols*rows, cudahostAllocPortable));


    m_bufferCPU = new CPUMatrix<float>(rows, cols, pinnedBuffer);
    
    

}

}}}


