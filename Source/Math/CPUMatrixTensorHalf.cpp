#include "stdafx.h"
#include "CPUMatrixTensorImpl.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template
void CPUMatrixTensorOpImpl(half beta, const CPUMatrix<half>& a, CPUMatrix<half>& o, half alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 2>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides);

template
void CPUMatrixTensorOpImpl(half beta, const CPUMatrix<half>& a, const CPUMatrix<half>& b, CPUMatrix<half>& o, half alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 3>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides);

template
void CPUMatrixTensorOpImpl(half beta, const CPUMatrix<half>& a, const CPUMatrix<half>& b, const CPUMatrix<half>& c, CPUMatrix<half>& o, half alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 4>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides);

template
void CPUMatrixTensorArgOpImpl(const CPUMatrix<half>& a, CPUMatrix<half>& o, ElementWiseOperator reductionOp,
    const array<size_t, 2>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides);

}}}