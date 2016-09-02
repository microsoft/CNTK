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

template <typename ElemType> SwapOutAction<ElemType>::SwapOutAction(Matrix<ElemType> *GPUbuffer)
{

        this->m_bufferCPU = NULL;
        this->m_bufferGPU = GPUbuffer;
        this->m_hasDoneInitialSwap = false;
		this->m_rows = this->m_bufferGPU->GetNumRows();
		this->m_cols = this->m_bufferGPU->GetNumCols();
		this->m_bytes = this->m_rows*this->m_cols*sizeof(ElemType);

#ifndef CPUONLY
        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreate(&stream));
        this->m_streamAsync = stream;

        // do we already have a pinned, that is page-locked buffer?
		if (!this->m_bufferCPU){ allocatePinnedBuffer(); }
#endif
}

template <typename ElemType> SwapOutAction<ElemType>::~SwapOutAction()
{
    ReleaseMemory();
}

template <typename ElemType> void SwapOutAction<ElemType>::BeginAction()
{
#ifndef CPUONLY
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
#endif
}


template <typename ElemType> void SwapOutAction<ElemType>::EndAction()
{
#ifndef CPUONLY
    CUDA_CALL(cudaStreamSynchronize(m_streamAsync));
    this->m_rows = this->m_bufferGPU->GetNumRows();
    this->m_cols = this->m_bufferGPU->GetNumCols();
    this->m_bytes = this->m_rows*this->m_cols*sizeof(ElemType);
    cout << "Swapped out: " << this->m_bufferGPU << ", " << this->m_bufferGPU->GetNumRows() << "x" << this->m_bufferGPU->GetNumCols() << ", " << this->m_rows*this->m_cols*sizeof(ElemType)/1024./1024./1024. << "GB" << endl;
    this->m_bufferGPU->Resize(0,0,0, false);
    m_hasDoneInitialSwap = true;
#endif

}


template <typename ElemType> void SwapOutAction<ElemType>::allocatePinnedBuffer()
{
#ifndef CPUONLY
    //cudaHostAllocPortable preservse the page-lock even across threads
    CUDA_CALL(cudaHostAlloc(&(this->m_bufferCPU), this->m_bytes, cudaHostAllocPortable));
#endif
}

template <typename ElemType> void SwapOutAction<ElemType>::ReleaseMemory()
{
#ifndef CPUONLY
    CUDA_CALL(cudaFreeHost(this->m_bufferCPU)); 
#endif
}

template class SwapOutAction<double>;
template class SwapOutAction<float>;

}}}