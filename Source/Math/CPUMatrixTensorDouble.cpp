#include "stdafx.h"
#include "CPUMatrixTensorImpl.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template
void CPUMatrixTensorOpImpl(double beta, const CPUMatrix<double>& a, CPUMatrix<double>& o, double alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 2>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides);

template
void CPUMatrixTensorOpImpl(double beta, const CPUMatrix<double>& a, const CPUMatrix<double>& b, CPUMatrix<double>& o, double alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 3>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides);

template
void CPUMatrixTensorOpImpl(double beta, const CPUMatrix<double>& a, const CPUMatrix<double>& b, const CPUMatrix<double>& c, CPUMatrix<double>& o, double alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 4>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides);

template
void CPUMatrixTensorArgOpImpl(const CPUMatrix<double>& a, CPUMatrix<double>& o, ElementWiseOperator reductionOp,
    const array<size_t, 2>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides);

}}}