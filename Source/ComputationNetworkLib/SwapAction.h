//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <CPUMatrix.h>
#include <GPUMatrix.h>

namespace Microsoft { namespace MSR { namespace CNTK {

template <typename ElemType>
class SwapAction
{

protected:
    Matrix<ElemType> *m_bufferGPU;
    ElemType *m_bufferCPU;
    int m_rows;
    int m_cols;
    size_t m_bytes;

public:
    ~SwapAction(){};
    virtual void BeginAction() = 0; // for starting asynchronous actions
    virtual void EndAction() = 0; // for synchronization and cleanup
    virtual void ReleaseMemory() = 0;
    Matrix<ElemType> *GetGPUMatrix(){ return m_bufferGPU; }
    ElemType *GetCPUMatrix(){ return m_bufferCPU; }

    int GetRows(){ return m_rows; };
    int GetCols(){ return m_cols; };

};

template class SwapAction<double>;
template class SwapAction<float>;

}}}
