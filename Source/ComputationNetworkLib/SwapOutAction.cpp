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

template SwapOutAction<double>::SwapOutAction(Matrix<double> *GPUbuffer);
template SwapOutAction<float>::SwapOutAction(Matrix<float> *GPUbuffer);
template <typename ElemType> SwapOutAction<ElemType>::SwapOutAction(Matrix<ElemType> *GPUbuffer)
{
        this->m_bufferCPU = NULL;
        this->m_bufferGPU = GPUbuffer;
        this->m_hasDoneInitialSwap = false;
        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreate(&stream));
        this->m_streamAsync = stream;
        this->m_rows = this->m_bufferGPU->GetNumRows();
        this->m_cols = this->m_bufferGPU->GetNumCols();
        this->m_bytes = this->m_rows*this->m_cols*sizeof(ElemType);

        // do we already have a pinned, that is page-locked buffer?
        if (!this->m_bufferCPU){ allocatePinnedBuffer(); }
}

template SwapOutAction<float>::~SwapOutAction();
template SwapOutAction<double>::~SwapOutAction();
template <typename ElemType> SwapOutAction<ElemType>::~SwapOutAction()
{
    ReleaseMemory();
}

template void SwapOutAction<double>::BeginAction();
template void SwapOutAction<float>::BeginAction();
template <typename ElemType> void SwapOutAction<ElemType>::BeginAction()
{
    // perform the actual asynchronous copy
    if(this->m_rows != this->m_bufferGPU->GetNumRows() ||
       this->m_cols != this->m_bufferGPU->GetNumCols())
       {
            if(this->m_bytes > 0)
                ReleaseMemory();

            this->m_rows = this->m_bufferGPU->GetNumRows();
            this->m_cols = this->m_bufferGPU->GetNumCols();
            this->m_bytes = this->m_rows*this->m_cols*sizeof(ElemType);
            allocatePinnedBuffer();
       }

    cout << "Begin swapping out: " << this->m_bufferGPU << ", " << this->m_bufferGPU->GetNumRows() << "x" << this->m_bufferGPU->GetNumCols() << ", " << this->m_rows*this->m_cols*sizeof(ElemType)/1024./1024./1024. << "GB" << endl;
    CUDA_CALL(cudaMemcpyAsync(this->m_bufferCPU, this->m_bufferGPU->Data(), this->m_bytes, cudaMemcpyDefault, this->m_streamAsync));
}



template void SwapOutAction<double>::EndAction();
template void SwapOutAction<float>::EndAction();
template <typename ElemType> void SwapOutAction<ElemType>::EndAction()
{
    CUDA_CALL(cudaStreamSynchronize(m_streamAsync));
    this->m_rows = this->m_bufferGPU->GetNumRows();
    this->m_cols = this->m_bufferGPU->GetNumCols();
    this->m_bytes = this->m_rows*this->m_cols*sizeof(ElemType);
    cout << "Swapped out: " << this->m_bufferGPU << ", " << this->m_bufferGPU->GetNumRows() << "x" << this->m_bufferGPU->GetNumCols() << ", " << this->m_rows*this->m_cols*sizeof(ElemType)/1024./1024./1024. << "GB" << endl;
    this->m_bufferGPU->Resize(0,0,0, false);
    m_hasDoneInitialSwap = true;

}


template void SwapOutAction<double>::allocatePinnedBuffer();
template void SwapOutAction<float>::allocatePinnedBuffer();
template <typename ElemType> void SwapOutAction<ElemType>::allocatePinnedBuffer()
{
    //cudaHostAllocPortable preservse the page-lock even across threads
    CUDA_CALL(cudaHostAlloc(&(this->m_bufferCPU), this->m_bytes, cudaHostAllocPortable));
}

template void SwapOutAction<float>::ReleaseMemory();
template void SwapOutAction<double>::ReleaseMemory();
template <typename ElemType> void SwapOutAction<ElemType>::ReleaseMemory(){ CUDA_CALL(cudaFreeHost(this->m_bufferCPU)); }

}}}


