//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "SyncAction.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class SwapOutAction;
class SwapInAction : public SyncAction
{

public:
    ~SwapInAction(){}
    SwapInAction(SwapOutAction *swpout, Matrix<float> *GPUBuffer);

    //implementation of abstract method
    void BeginAction();
    void EndAction();
    void ReleaseMemory(){};
private:
    cudaStream_t m_swapInStream;
    bool m_isSwappingToGPU;
    SwapOutAction *m_swpout;
    int m_batchSize;

};
}}}
