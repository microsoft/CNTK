//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "SyncAction.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <typename ElemType> class SwapOutAction;

template <typename ElemType> 
class SwapInAction : public SyncAction<ElemType>
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
    int m_batchSize;

};

template class SwapInAction<double>;
template class SwapInAction<float>;

}}}
