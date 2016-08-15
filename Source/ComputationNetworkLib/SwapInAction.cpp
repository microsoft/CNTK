//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "SwapInAction.h"
#include "SwapOutAction.h"
#include "GPUMatrix.h"
#include <iostream>

#ifndef CPUONLY
    #include <cuda.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

using std::cout;
using std::endl;

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
   if(!this->m_swpout->m_hasDoneInitialSwap){ return; }

   this->m_bufferGPU->Resize(this->m_swpout->GetRows(),this->m_swpout->GetCols(), 0, false);
   size_t bytes = this->m_swpout->GetRows()*this->m_swpout->GetCols()*sizeof(ElemType);

   //cudaPointerAttributes bla;
   //ElemType *ptr = this->m_bufferGPU->FullData();
   //ElemType *ptr;
   //CUDA_CALL(cudaMalloc((void**)&ptr, bytes));
   //ElemType *data = this->m_bufferGPU->Data();
   //CUDA_CALL(cudaMemcpy(&(data[0]), ptr, bytes, cudaMemcpyDefault));
   //cout << ptr << data << endl;
   //cout << ptr << this->m_bufferGPU->FullData() << endl;
   //CUDA_CALL(cudaMalloc(&ptr, bytes));
   //cout << ptr << this->m_bufferGPU->FullData() << endl;
   //cout << ptr << this->m_bufferGPU->FullData() << endl;
   //CUDA_CALL(cudaPointerGetAttributes(&bla, this->m_bufferGPU->FullData()));
   //cout << bla.devicePointer << " " << bla.hostPointer << endl;
   //CUDA_CALL(cudaPointerGetAttributes(&bla, ptr));
   //cout << bla.devicePointer << " " << bla.hostPointer << endl;


   CUDA_CALL(cudaMemcpyAsync(this->m_bufferGPU->Data(), this->m_swpout->GetCPUMatrix(), bytes, cudaMemcpyDefault, this->m_swapInStream));
   cout << "begin swapping in" << endl;
}


template void SwapInAction<double>::EndAction();
template void SwapInAction<float>::EndAction();
template <typename ElemType> void SwapInAction<ElemType>::EndAction()
{
    if(!this->m_swpout->m_hasDoneInitialSwap){ return; }
    CUDA_CALL(cudaStreamSynchronize(this->m_swapInStream));
    cout << "Swapped in: " << this->m_bufferGPU << ", " << this->m_rows*this->m_cols*sizeof(ElemType)/1024./1024./1024. << "GB" << endl;
}


}}}
