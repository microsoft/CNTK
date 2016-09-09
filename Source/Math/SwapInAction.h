//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "SwapAction.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <typename ElemType> class SwapOutAction;

template <typename ElemType> 
class SwapInAction : public SwapAction<ElemType>
{

public:
    ~SwapInAction(){}
    SwapInAction(SwapOutAction<ElemType> *swpout, Matrix<ElemType> *GPUBuffer);

    //implementation of abstract methods
    void BeginAction();
    void EndAction();
    void ReleaseMemory(){};
private:
    cudaStream_t m_swapInStream;
    SwapOutAction<ElemType> *m_swpout;
};

}}}
