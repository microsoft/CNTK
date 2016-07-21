//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "SwapOutAction.h"
#include <iostream>
#include <string>

#ifndef CPUONLY
	#include <cuda_runtime.h>
#endif


namespace Microsoft { namespace MSR { namespace CNTK {

using std::cout;
using std::endl;

SwapOutAction::SwapOutAction(Matrix<float> *GPUbuffer)
{
        m_bufferCPU = NULL;
        m_bufferGPU = GPUbuffer;
        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreate(&stream));
        m_streamAsync = stream;
        m_rows = m_bufferGPU->GetNumRows();
        m_cols = m_bufferGPU->GetNumCols();
        m_bytes = m_rows*m_cols*sizeof(float);

        // do we already have a pinned, that is page-locked buffer?
        if (!m_bufferCPU){ allocatePinnedBuffer(); }
}

SwapOutAction::~SwapOutAction(){ ReleaseMemory(); }

void SwapOutAction::BeginAction()
{
    // perform the actual asynchronous copy
    CUDA_CALL(cudaMemcpyAsync(m_bufferCPU, m_bufferGPU->Data(), m_bytes, cudaMemcpyDefault, m_streamAsync));
}

void SwapOutAction::EndAction()
{
    CUDA_CALL(cudaStreamSynchronize(m_streamAsync));
    m_rows = m_bufferGPU->GetNumRows();
    m_cols = m_bufferGPU->GetNumCols();
}


void SwapOutAction::allocatePinnedBuffer()
{
    //cudaHostAllocPortable preservse the page-lock even across threads
    CUDA_CALL(cudaHostAlloc(&m_bufferCPU, m_bytes, cudaHostAllocPortable));
}

void SwapOutAction::ReleaseMemory(){ CUDA_CALL(cudaFreeHost(m_bufferCPU)); }

}}}


