//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "SwapInAction.h"
#include "SwapOutAction.h"
#include "GPUMatrix.h"

#ifndef CPUONLY
    #include <cuda.h>
#endif



namespace Microsoft { namespace MSR { namespace CNTK {

template SwapInAction<float>::SwapInAction(SwapOutAction<float> *swpout, Matrix<float> *GPUBuffer);
template SwapInAction<double>::SwapInAction(SwapOutAction<double> *swpout, Matrix<double> *GPUBuffer);
template <typename ElemType> SwapInAction<ElemType>::SwapInAction(SwapOutAction<ElemType> *swpout, Matrix<ElemType> *GPUBuffer)
{
    this->m_bufferCPU = swpout->GetCPUMatrix();
    this->m_bufferGPU = GPUBuffer;
    this->m_swpout = swpout;

    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    this->m_swapInStream = stream;
    this->m_rows = this->m_bufferGPU->GetNumRows();
    this->m_cols = this->m_bufferGPU->GetNumCols();
    this->m_bytes = this->m_rows*this->m_cols*sizeof(ElemType);
}

 

template void SwapInAction<float>::BeginAction();
template void SwapInAction<double>::BeginAction();
template <typename ElemType> void SwapInAction<ElemType>::BeginAction()
{
   this->m_bufferGPU->Resize(this->m_swpout->GetRows(),this->m_swpout->GetCols());
   //size_t bytes = this->m_swpout->GetRows()*this->m_swpout->GetCols()*sizeof(ElemType);
   //ElemType *ptr = this->m_bufferGPU->Data();
   //CUDA_CALL(cudaMalloc((void**)&ptr, bytes));
   CUDA_CALL(cudaMemcpyAsync(this->m_bufferGPU->Data(), this->m_bufferCPU, this->m_bytes, cudaMemcpyDefault, this->m_swapInStream));
}


template void SwapInAction<double>::EndAction();
template void SwapInAction<float>::EndAction();
template <typename ElemType> void SwapInAction<ElemType>::EndAction(){ CUDA_CALL(cudaStreamSynchronize(this->m_swapInStream)); }


}}}
