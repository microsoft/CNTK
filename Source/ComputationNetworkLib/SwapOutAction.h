//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "SyncAction.h"
#include "GPUMatrix.h"

#ifndef ONLYCPU
    #include <cuda.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

class SwapOutAction : public SyncAction
{

public:
    ~SwapOutAction(){}
    SwapOutAction(Matrix<float> *GPUbuffer)
    {
        m_bufferCPU = NULL;
        m_bufferGPU = GPUbuffer;
        m_isAsynchronous = false;
        cudaStream_t stream;
        CUDA_CALL(cudaStreamCreate(&stream));
        m_streamAsync = stream;

        // do we already have a pinned, that is page-locked buffer?
        if (!m_bufferCPU){ allocatePinnedBuffer(); }
    }

    //implementation of abstract method
    void BeginAction();
    void endAction();
    cudaStream_t GetSwapSteam(){ return m_streamAsync; }

private:
    cudaStream_t m_streamAsync; 
    void allocatePinnedBuffer();

};

}}}
