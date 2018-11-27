#include "stdafx.h"

#ifdef USE_MKL

#include "CPUMatrixTensorImpl.h"
#include "mkl_cblas.h"
#include "mkl_vml.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<>
bool CPUMatrixSpecialUnaryTensorOpImpl<float>(float beta, const CPUMatrix<float>& a, CPUMatrix<float>& o, float alpha, ElementWiseOperator op, ElementWiseOperator /*reductionOp*/,
    const array<size_t, 2>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& /*reducingStrides*/)
{
    if (alpha == 1.0f && beta == 0.0f && // for inference
        reducingOpDims.size() == 0 && // no reduction
        regularStrides[0] == regularStrides[1]) // input/output have the same strides
    {
        // check if it is elementwise operation with 1:1 input/output mapping and no gap
        size_t count = 1;
        for (int rank = 0; rank < regularOpDims.size(); ++ rank)
        {
            // 0 stride can only be in the last rank
            if (regularStrides[0][rank] == 0 && rank != regularStrides[0].size() - 1)
                return false;

            // if not continuous in memory, don't optimize
            if ((ptrdiff_t)count != regularStrides[0][rank] || regularStrides[0][rank] == 0)
                return false;

            count *= regularOpDims[rank];
        }

        float* pA = a.Data() + offsets[0];
        float* pO = o.Data() + offsets[1];

        switch (op)
        {
        case ElementWiseOperator::opLinearRectifier:
            if (pA != pO)
            {
                vsAbs((int)count, pA, pO);
                cblas_saxpby((int)count, 0.5f, pA, 1, 0.5f, pO, 1); // o = (a + abs(a))/2
                return true;
            }
        }
    }
    return false;
}

template<>
bool CPUMatrixSpecialBinaryTensorOpImpl<float>(float beta, const CPUMatrix<float>& a, const CPUMatrix<float>& b, CPUMatrix<float>& o, float alpha, ElementWiseOperator op, ElementWiseOperator /*reductionOp*/,
    const array<size_t, 3>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& /*reducingStrides*/)
{
    if (alpha == 1.0f && beta == 0.0f && // for inference
        reducingOpDims.size() == 0 && // no reduction
        (regularStrides[0] == regularStrides[2] ||
         regularStrides[1] == regularStrides[2])) // one of the inputs has same strides as output
    {
        // only support simple broadcasting case
        if (regularStrides[0].size() != regularStrides[1].size())
        {
            for (int rank = 0; rank < std::min(regularStrides[0].size(), regularStrides[1].size()); ++rank)
            {
                if (regularStrides[0][rank] != regularStrides[1][rank])
                    return false;
            }
        }

        // MKL based optimization on scalar/vector, vector/vector, and matrix/vector operations

        size_t elementCount[3] = { 1, 1, 1 }; // element count for a/b/o
        for (int rank = 0; rank < regularOpDims.size(); ++ rank)
        {
            for (int iOp = 0; iOp < _countof(elementCount); ++ iOp)
            {
                // 0 stride can only be in the last rank
                if (regularStrides[iOp][rank] == 0 && rank != regularStrides[iOp].size() - 1)
                    return false;

                if (rank >= regularStrides[iOp].size() || regularStrides[iOp][rank] == 0) continue;

                // if not continuous in memory, don't optimize
                if (regularStrides[iOp][rank] != (ptrdiff_t)elementCount[iOp])
                    return false;

                elementCount[iOp] *= regularOpDims[rank];
            }
        }
        size_t aN = elementCount[0];
        size_t bN = elementCount[1];
        size_t oN = elementCount[2];
        float* pA = a.Data() + offsets[0];
        float* pB = b.Data() + offsets[1];
        float* pO = o.Data() + offsets[2];
        int count = (int)oN;

        // scalar/vector
        if ((aN == oN && bN == 1) || (bN == oN && aN == 1))
        {
            float scalar = (aN == 1 ? pA[0] : pB[0]);
            float* input = (aN == 1 ? pB : pA);

            if (input != pO)
                memcpy(pO, input, count * sizeof(float));

            switch (op)
            {
            case ElementWiseOperator::opElementwiseProduct:
                cblas_sscal(count, scalar, pO, 1);
                return true;
            case ElementWiseOperator::opSum:
                cblas_saxpby(count, 1.0f, &scalar, 0, 1.0f, pO, 1);
                return true;
            case ElementWiseOperator::opDifference:
                if (input == pA)
                    cblas_saxpby(count, -1.0f, &scalar, 0, 1.0f, pO, 1);
                else
                    cblas_saxpby(count, 1.0f, &scalar, 0, -1.0f, pO, 1);
                return true;
            }
        }
        // vector/vector (elementwise 1:1)
        else if (aN == oN && bN == oN)
        {
            // elementwise operation with no broadcast/reduction
            switch (op)
            {
            case ElementWiseOperator::opSum:
                vsAdd(count, pA, pB, pO);
                return true;
            case ElementWiseOperator::opElementwiseProduct:
                vsMul(count, pA, pB, pO);
                return true;
            case ElementWiseOperator::opDifference:
                vsSub(count, pA, pB, pO);
                return true;
            }
        }
        // vector/matrix, i.e. plus/multiply parameter
        else if (std::max(aN, bN) == oN)
        {
            float* pMat = (aN < bN ? pB : pA);
            float* pVec = (aN < bN ? pA : pB);
            int vecN = (int)std::min(aN, bN);
            int numVec = (int)(oN / vecN);
            switch (op)
            {
            case ElementWiseOperator::opSum:
                for (int i = 0; i < numVec; ++i)
                {
                    vsAdd(vecN, pMat + i * vecN, pVec, pO + i * vecN);
                }
                return true;
            case ElementWiseOperator::opElementwiseProduct:
                for (int i = 0; i < numVec; ++i)
                {
                    vsMul(vecN, pMat + i * vecN, pVec, pO + i * vecN);
                }
                return true;
            }
        }
    }
    return false;
}

template<>
bool CPUMatrixSpecialTernaryTensorOpImpl<float>(float /*beta*/, const CPUMatrix<float>& /*a*/, const CPUMatrix<float>& /*b*/, const CPUMatrix<float>& /*c*/, CPUMatrix<float>& /*o*/, float /*alpha*/, ElementWiseOperator /*op*/, ElementWiseOperator /*reductionOp*/,
    const array<size_t, 4>& /*offsets*/,
    const SmallVector<size_t>& /*regularOpDims*/, const array<SmallVector<ptrdiff_t>, 4>& /*regularStrides*/,
    const SmallVector<size_t>& /*reducingOpDims*/, const array<SmallVector<ptrdiff_t>, 4>& /*reducingStrides*/)
{
    return false;
}

template<>
bool CPUMatrixSpecialUnaryTensorOpImpl<double>(double, const CPUMatrix<double>&, CPUMatrix<double>&, double, ElementWiseOperator, ElementWiseOperator,
    const array<size_t, 2>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 2>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 2>&)
{
    return false;
}

template<>
bool CPUMatrixSpecialBinaryTensorOpImpl<double>(double, const CPUMatrix<double>&, const CPUMatrix<double>&, CPUMatrix<double>&, double, ElementWiseOperator, ElementWiseOperator,
    const array<size_t, 3>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 3>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 3>&)
{
    return false;
}

template<>
bool CPUMatrixSpecialTernaryTensorOpImpl<double>(double, const CPUMatrix<double>&, const CPUMatrix<double>&, const CPUMatrix<double>&, CPUMatrix<double>&, double, ElementWiseOperator, ElementWiseOperator,
    const array<size_t, 4>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 4>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 4>&)
{
    return false;
}

template<>
bool CPUMatrixSpecialUnaryTensorOpImpl<half>(half, const CPUMatrix<half>&, CPUMatrix<half>&, half, ElementWiseOperator, ElementWiseOperator,
    const array<size_t, 2>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 2>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 2>&)
{
    return false;
}

template<>
bool CPUMatrixSpecialBinaryTensorOpImpl<half>(half, const CPUMatrix<half>&, const CPUMatrix<half>&, CPUMatrix<half>&, half, ElementWiseOperator, ElementWiseOperator,
    const array<size_t, 3>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 3>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 3>&)
{
    return false;
}

template<>
bool CPUMatrixSpecialTernaryTensorOpImpl<half>(half, const CPUMatrix<half>&, const CPUMatrix<half>&, const CPUMatrix<half>&, CPUMatrix<half>&, half, ElementWiseOperator, ElementWiseOperator,
    const array<size_t, 4>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 4>&,
    const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, 4>&)
{
    return false;
}

}}}

#endif