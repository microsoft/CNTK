//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "GPUMatrix.h"
#include "Matrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <typename ElemType> class Matrix;

template <typename ElemType>
class SwapAction
{

protected:
    Matrix<ElemType> *m_bufferGPU;
    ElemType *m_bufferCPU;
	size_t m_rows;
	size_t m_cols;
    size_t m_bytes;

public:
    ~SwapAction(){};
    virtual void BeginAction() = 0; // for starting asynchronous actions
    virtual void EndAction() = 0; // for synchronization and cleanup
    virtual void ReleaseMemory() = 0;
    Matrix<ElemType> *GetGPUMatrix(){ return m_bufferGPU; }
    ElemType *GetCPUMatrix(){ return m_bufferCPU; }

    size_t GetRows(){ return m_rows; };
	size_t GetCols(){ return m_cols; };

};

template class SwapAction<double>;
template class SwapAction<float>;

}}}
