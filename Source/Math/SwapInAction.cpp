//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "stdafx.h"
#include "SwapInAction.h"
#include "SwapOutAction.h"
#include <iostream>

#ifndef CPUONLY
    #include <cuda.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

using std::cout;
using std::endl;

template <typename ElemType> SwapInAction<ElemType>::SwapInAction(SwapOutAction<ElemType> *swpout, Matrix<ElemType> *GPUBuffer)
{

	this->m_bufferGPU = GPUBuffer;
	this->m_swpout = swpout;
	this->m_bufferCPU = swpout->GetCPUMatrix();
	this->m_rows = this->m_bufferGPU->GetNumRows();
	this->m_cols = this->m_bufferGPU->GetNumCols();
	this->m_bytes = this->m_rows*this->m_cols*sizeof(ElemType);

#ifndef CPUONLY
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    this->m_swapInStream = stream;
#endif
}


template <typename ElemType> void SwapInAction<ElemType>::BeginAction()
{
#ifndef CPUONLY
   if(!this->m_swpout->m_hasDoneInitialSwap){ return; }
   if(!this->m_swpout->m_isSwappedOut){ return; }

   this->m_bufferGPU->Resize(this->m_swpout->GetRows(),this->m_swpout->GetCols(), 0, false);
   size_t bytes = this->m_swpout->GetRows()*this->m_swpout->GetCols()*sizeof(ElemType);

   CUDA_CALL(cudaMemcpyAsync(this->m_bufferGPU->Data(), this->m_swpout->GetCPUMatrix(), bytes, cudaMemcpyDefault, this->m_swapInStream));
#endif
}


template <typename ElemType> void SwapInAction<ElemType>::EndAction()
{
#ifndef CPUONLY
   if(!this->m_swpout->m_hasDoneInitialSwap){ return; }
   if(!this->m_swpout->m_isSwappedOut){ return; }

    CUDA_CALL(cudaStreamSynchronize(this->m_swapInStream));
    cout << "Swapped in: " << this->m_bufferGPU << ", " << this->m_bufferGPU->BufferSize()/1024./1024./1024. << "GB" << endl;
   this->m_swpout->m_isSwappedOut = false;
#endif
}

template class SwapInAction<double>;
template class SwapInAction<float>;

}}}
