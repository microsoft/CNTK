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

SwapInAction::SwapInAction(SwapOutAction *swpout, Matrix<float> *GPUBuffer)
{
    m_bufferCPU = swpout->GetCPUMatrix();
    m_bufferGPU = GPUBuffer;
    m_swpout = swpout;

    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));
    m_swapInStream = stream;
    m_rows = m_bufferGPU->GetNumRows();
    m_cols = m_bufferGPU->GetNumCols();
    m_bytes = m_rows*m_cols*sizeof(float);
}

 

void SwapInAction::BeginAction()
{
   m_bufferGPU->Resize(m_swpout->GetRows(),m_swpout->GetCols());
   CUDA_CALL(cudaMemcpyAsync(m_bufferGPU->Data(), m_bufferCPU, m_bytes, cudaMemcpyDefault, m_swapInStream));
}


void SwapInAction::EndAction(){ CUDA_CALL(cudaStreamSynchronize(m_swapInStream)); }


}}}
