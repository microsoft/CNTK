//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CPUMatrixImpl.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    int MATH_API TracingGPUMemoryAllocator::m_traceLevel = 0;

    void TracingGPUMemoryAllocator::SetTraceLevel(int traceLevel)
    {
        m_traceLevel = traceLevel;
    }

    bool TracingGPUMemoryAllocator::IsTraceEnabled()
    {
        return (m_traceLevel > 0);
    }

    // explicit instantiations, due to CPUMatrix being too big and causing VS2015 cl crash.
    template class MATH_API CPUMatrix<float>;

    template CPUMatrix<float>& CPUMatrix<float>::DoScatterColumnsOf<float>(float beta, const CPUMatrix<float>& idx, const CPUMatrix<float>& a, float alpha);
    template CPUMatrix<float>& CPUMatrix<float>::DoScatterColumnsOf<int  >(float beta, const CPUMatrix<int  >& idx, const CPUMatrix<float>& a, float alpha);
}}}
