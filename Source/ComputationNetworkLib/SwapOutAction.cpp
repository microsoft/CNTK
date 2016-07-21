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


void PrintPtrAttributes2(float *ptr)
{
    cudaPointerAttributes att;
    cudaPointerGetAttributes(&att,ptr);
    cout << "memtype: " << att.memoryType << endl;
    cout << "device: " << att.device << endl;
    cout << "host: " << att.hostPointer << endl;
    cout << "device ptr: " << att.devicePointer << endl;
}

SwapOutAction::SwapOutAction(Matrix<float> *GPUbuffer)
{
        m_bufferCPU = NULL;
        m_bufferGPU = GPUbuffer;
        m_isAsynchronous = false;
        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreate(&stream));
        m_streamAsync = stream;
        m_isSwapping = false;
        m_rows = m_bufferGPU->GetNumRows();
        m_cols = m_bufferGPU->GetNumCols();
        m_bytes = m_rows*m_cols*sizeof(float);
        m_timer = CUDATimer();
        m_syncCounter = 0;
        //cout << m_rows << "x" << m_cols << endl;


        // do we already have a pinned, that is page-locked buffer?
        if (!m_bufferCPU){ allocatePinnedBuffer(); }
    }
SwapOutAction::~SwapOutAction()
{
    // TODO: can we check if the memory was release before
    ReleaseMemory();
}

void SwapOutAction::BeginAction()
{
    if(m_isSwapping)
    {
        cout << "Warning: Overlapping swap-outs detected!" << endl;
        //EndAction();
    }
    // perform the actual asynchronous copy
    CUDA_CALL(cudaMemcpyAsync(m_bufferCPU, m_bufferGPU->Data(), m_bytes, cudaMemcpyDefault, m_streamAsync));
    m_isSwapping = true;
}

void SwapOutAction::EndAction()
{
    //m_timer.tick();
    CUDA_CALL(cudaStreamSynchronize(m_streamAsync));
    //m_timer.tick();

    m_isSwapping = false;
    m_rows = m_bufferGPU->GetNumRows();
    m_cols = m_bufferGPU->GetNumCols();
    //cout << "swap out: " << m_rows << "x" << m_cols << endl;
    //m_timer.tick("resize");
    //m_bufferGPU->Resize(0,0,0,false);
    //cout << "free " << endl;
    //float *gpudata = m_bufferGPU->Data();
    //CUDA_CALL(cudaFree(gpudata));
    //m_timer.tick("resize");

    //m_syncCounter++;

    if(m_syncCounter > 1000)
    {
        //cout << m_timer.tock()/1000.0f << endl;
        cout << "resize: " << m_timer.tock("resize")/1000.0f << endl;
        m_syncCounter = 0;
    }
}


void SwapOutAction::allocatePinnedBuffer()
{
    size_t cols = m_bufferGPU->GetNumCols();
    size_t rows = m_bufferGPU->GetNumRows();
    size_t bytes = cols*rows*sizeof(float);

    float *pinnedBuffer;
    //cudaHostAllocPortable preservse the page-lock even across threads
    CUDA_CALL(cudaHostAlloc(&pinnedBuffer, bytes, cudaHostAllocPortable));
    m_bufferCPU = pinnedBuffer;
}

void SwapOutAction::ReleaseMemory()
{
    CUDA_CALL(cudaFreeHost(m_bufferCPU));
}

}}}


