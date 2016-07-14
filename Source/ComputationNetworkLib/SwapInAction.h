//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "SyncAction.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class SwapInAction : public SyncAction
{

public:

    ~SwapInAction(){}
    SwapInAction(float *CPUBuffer, Matrix<float> *GPUBuffer)
    {
        m_bufferCPU = CPUBuffer;
        m_bufferGPU = GPUBuffer;
        m_isAsynchronous = true;

        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreate(&stream));
        m_swapInStream = stream;
        m_isSwappingToGPU = false;

        m_rows = m_bufferGPU->GetNumRows();
        m_cols = m_bufferGPU->GetNumCols();
        m_bytes = m_rows*m_cols*sizeof(float);
    }

    //implementation of abstract method
    void BeginAction();
    void EndAction();
    void ReleaseMemory(){};

private:
    cudaStream_t m_swapInStream;
    bool m_isSwappingToGPU;

};
}}}
