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
    }

    //implementation of abstract method
    void BeginAction();
    void EndAction();

private:
    cudaStream_t m_swapInStream;
    bool m_isSwappingToGPU;

};
}}}
