// Move some files out of CPUMatrixImpl.h to prevent compiler crash on out-of-heap

#include "CPUMatrix.h"
#include "TensorOps.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// =======================================================================
// TensorView support
// =======================================================================

// To save time, this makes extensive use of templates and macros.

// -----------------------------------------------------------------------
// function to compute the value for a given output location (perform reduction if needed)
// -----------------------------------------------------------------------

// perform loop over reduction index m
// This function is declared inside a wrapper struct to allow partial specialization (m = -1).
template <class ElemType, typename OPFN, typename ReductionOp, size_t N, int m>
struct TensorOpReduction
{
    // reduction case (non-reduction case is specialized)
    static inline ElemType Loop(array<ElemType*, N> pointers, const OPFN& opfn, const ReductionOp& reductionOp,
                                const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
    {
        array<ptrdiff_t, N - 1> strides;   // N-1 because last one is the result pointer, which is unused in reduction
        for (size_t i = 0; i < N - 1; i++) // N = a small constant, this will be unrolled
            strides[i] = reducingStrides[i][(size_t) m];

        double aggregate = TensorOpReduction<ElemType, OPFN, ReductionOp, N, m - 1>::Loop(pointers, opfn, reductionOp, reducingOpDims, reducingStrides);
        for (size_t dim = reducingOpDims[(size_t)m] - 1; dim-- > 0;)
        {
            // advance the pointers
            for (size_t i = 0; i < N - 1; i++)
                pointers[i] += strides[i]; // note: last pointer (result) is unused and untouched here

            // need to descend into one loop deeper
            aggregate = reductionOp(aggregate, TensorOpReduction<ElemType, OPFN, ReductionOp, N, m - 1>::Loop(pointers, opfn, reductionOp, reducingOpDims, reducingStrides));
        }
        // Actually it would be nicer to return double but we keep ElementType so that test don't return different numbers than previous implementation.
        return static_cast<ElemType>(aggregate);
    }
};

// perform loop over reduction index m
// This is the specialized version for m = -1, which terminates the recursion.
template <class ElemType, typename OPFN, typename ReductionOp, size_t N>
struct TensorOpReduction<ElemType, OPFN, ReductionOp, N, -1>
{
    static inline ElemType Loop(array<ElemType*, N> pointers, const OPFN& opfn, const ReductionOp& /*reductionOp*/,
                                const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, N>&)
    {
        return opfn(pointers); // finally we are doing some work!!!
    }
};

// perform loop over reduction index m, while keeping track of the number of elements and their corresponding indices.
// This function is declared inside a wrapper struct to allow partial specialization (m = -1).
template <class ElemType, size_t N, int m>
struct TensorArgOpReduction
{
    static inline std::pair<ElemType, size_t> ReduceAll(array<ElemType*, N> pointers, const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides,
        ElementWiseOperator reductionOp)
    {
        size_t counter = 0;
        size_t index = 0;
        ElemType val = (ElemType)0;

        switch (reducingOpDims.size())
        {
        case 3:
            val = TensorArgOpReduction<ElemType, N, 2>::Loop(pointers, reducingOpDims, reducingStrides, reductionOp, counter, index);
            break;
        case 2:
            val = TensorArgOpReduction<ElemType, N, 1>::Loop(pointers, reducingOpDims, reducingStrides, reductionOp, counter, index);
            break;
        case 1:
            val = TensorArgOpReduction<ElemType, N, 0>::Loop(pointers, reducingOpDims, reducingStrides, reductionOp, counter, index);
            break;
        case 0:
            val = TensorArgOpReduction<ElemType, N, -1>::Loop(pointers, reducingOpDims, reducingStrides, reductionOp, counter, index);
            break;
        default:
            LogicError("TensorOp: %d non-flattened input dimensions are not supported.", (int)reducingOpDims.size());
        }

        return make_pair(val, index);
    }

    // reduction case (non-reduction case is specialized)
    static inline ElemType Loop(array<ElemType*, N> pointers, const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides,
                                ElementWiseOperator reductionOp, size_t& counter, size_t& index)
    {
        array<ptrdiff_t, N - 1> strides;   // N-1 because last one is the result pointer, which is unused in reduction
        for (size_t i = 0; i < N - 1; i++) // N = a small constant, this will be unrolled
            strides[i] = reducingStrides[i][(size_t)m];

        ElemType aggregate = TensorArgOpReduction<ElemType, N, m - 1>::Loop(pointers, reducingOpDims, reducingStrides, reductionOp, counter, index);
        for (size_t dim = reducingOpDims[(size_t)m] - 1; dim-- > 0;)
        {
            // advance the pointers
            for (size_t i = 0; i < N - 1; i++)
                pointers[i] += strides[i]; // note: last pointer (result) is unused and untouched here

            ElemType val = TensorArgOpReduction<ElemType, N, m - 1>::Loop(pointers, reducingOpDims, reducingStrides, reductionOp, counter, index);

            bool update = false;
            switch (reductionOp)
            {
            case ElementWiseOperator::opArgmin:
                update = (aggregate > val);
                break;
            case ElementWiseOperator::opArgmax:
                update = (aggregate < val);
                break;
            }

            if (update)
            {
                aggregate = val;
                index = counter - 1;
            }
        }

        return aggregate;
    }
};

// perform loop over reduction index m
// This is the specialized version for m = -1, which terminates the recursion.
template <class ElemType, size_t N>
struct TensorArgOpReduction<ElemType, N, -1>
{
    static inline ElemType Loop(array<ElemType*, N> pointers,
        const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, N>&, ElementWiseOperator /*reductionOp*/, size_t& counter, size_t& /*index*/)
    {
        counter++;
        return *pointers[0]; // finally we are doing some work!!!
    }
};

// -----------------------------------------------------------------------
// perform loop over regular index k for N-nary operations (N counting the output)
// -----------------------------------------------------------------------

// perform loop over regular index k and reducing index m for N operands (counting the output)
template <class ElemType, typename OPFN, typename ReductionOp, size_t N, bool vectorizable, int m, int k>
struct TensorOpIteration
{
    static inline void Loop(ElemType beta, array<ElemType*, N> pointers, ElemType alpha, const OPFN& opfn, const ReductionOp& reductionOp,
                            const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrides,
                            const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
    {
        // non-scalar case: still nested result loops left
        array<ptrdiff_t, N> strides;
        for (size_t i = 0; i < N; i++) // N = a small constant, this will be unrolled
            strides[i] = regularStrides[i][(size_t) k];
        for (size_t dim = regularOpDims[(size_t) k]; dim-- > 0;)
        {
            // need to descend into one loop deeper
            TensorOpIteration<ElemType, OPFN, ReductionOp, N, vectorizable, m, k - 1>::Loop(beta, pointers, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
            // advance the pointers
            for (size_t i = 0; i < N; i++)
                pointers[i] += strides[i];
        }
    }
};

// Special version for innermost loop with strides all being 1 and no further reduction. Compiler can use SSE.
// This is a very common case, e.g. adding vectors or computing the Sigmoid.
template <class ElemType, typename OPFN, typename ReductionOp>
struct TensorOpIteration<ElemType, OPFN, ReductionOp, 3, true /*vectorizable*/, -1 /*no reduction*/, 0 /*innermost loop*/>
{
    static inline void Loop(ElemType beta, array<ElemType*, 3> pointers, ElemType alpha, const OPFN& opfn, const ReductionOp& reductionOp,
                            const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
                            const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides)
    {
        ElemType* pa = pointers[0];
        ElemType* pb = pointers[1];
        ElemType* pc = pointers[2];
        size_t K = regularOpDims[0];
        // special-case beta and alpha to allow the compiler to short-circuit it
        if (beta != 0)
#pragma omp parallel for
            for (int k = 0; k < (int) K; k++)
                TensorOpIteration<ElemType, OPFN, ReductionOp, 3, true /*vectorizable*/, -1 /*no reduction*/, -1 /*scalar*/>::Loop(beta, array<ElemType*, 3>{pa + k, pb + k, pc + k}, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
        else if (alpha != 1)
#pragma omp parallel for
            for (int k = 0; k < (int) K; k++)
                TensorOpIteration<ElemType, OPFN, ReductionOp, 3, true /*vectorizable*/, -1 /*no reduction*/, -1 /*scalar*/>::Loop(0, array<ElemType*, 3>{pa + k, pb + k, pc + k}, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
        else
#pragma omp parallel for
            for (int k = 0; k < (int) K; k++)
                TensorOpIteration<ElemType, OPFN, ReductionOp, 3, true /*vectorizable*/, -1 /*no reduction*/, -1 /*scalar*/>::Loop(0, array<ElemType*, 3>{pa + k, pb + k, pc + k}, 1, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
        // TODO: According to Amit, the VS compiler is not able to vectorize into lambdas. Solution: change the lambda to take an N, or to implement the loop inside (with 1 element by default).
        // TODO: The signedness of k (required for omp) causes an extra sign-extend.
        // TODO: OMP adds LOTS of overhead. Do we need a guard, a min size when to use it?
    }
};
// and unary
template <class ElemType, typename OPFN, typename ReductionOp>
struct TensorOpIteration<ElemType, OPFN, ReductionOp, 2, true /*vectorizable*/, -1 /*no reduction*/, 0 /*innermost loop*/>
{
    static inline void Loop(ElemType beta, array<ElemType*, 2> pointers, ElemType alpha, const OPFN& opfn, const ReductionOp& reductionOp,
                            const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                            const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
    {
        ElemType* pa = pointers[0];
        ElemType* pb = pointers[1];
        size_t K = regularOpDims[0];
        // special-case beta and alpha to allow the compiler to short-circuit it
        if (beta != 0)
#pragma omp parallel for
            for (int k = 0; k < (int) K; k++)
                TensorOpIteration<ElemType, OPFN, ReductionOp, 2, true /*vectorizable*/, -1 /*no reduction*/, -1 /*scalar*/>::Loop(beta, array<ElemType*, 2>{pa + k, pb + k}, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
        else if (alpha != 1)
#pragma omp parallel for
            for (int k = 0; k < (int) K; k++)
                TensorOpIteration<ElemType, OPFN, ReductionOp, 2, true /*vectorizable*/, -1 /*no reduction*/, -1 /*scalar*/>::Loop(0, array<ElemType*, 2>{pa + k, pb + k}, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
        else
#pragma omp parallel for
            for (int k = 0; k < (int) K; k++)
                TensorOpIteration<ElemType, OPFN, ReductionOp, 2, true /*vectorizable*/, -1 /*no reduction*/, -1 /*scalar*/>::Loop(0, array<ElemType*, 2>{pa + k, pb + k}, 1, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    }
};

template <class ElemType, typename OPFN, typename ReductionOp, size_t N, bool vectorizable, int m>
struct TensorOpIteration<ElemType, OPFN, ReductionOp, N, vectorizable, m, -1>
{
    static inline void Loop(ElemType beta, array<ElemType*, N> pointers, ElemType alpha, const OPFN& opfn, const ReductionOp& reductionOp,
                            const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, N>&,
                            const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
    {
        // we are at element level for the result: perform the op (there may still be reduction)
        ElemType val = TensorOpReduction<ElemType, OPFN, ReductionOp, N, m>::Loop(pointers, opfn, reductionOp, reducingOpDims, reducingStrides);
        // scale
        val *= alpha;
        // combine with previous value in target matrix, then write it out
        auto* pout = pointers.back();
        if (beta != 0)
            val += beta * *pout;
        // save
        *pout = val;
        return;
    }
};

// perform loop over regular index k and reducing index m for N operands (counting the output), the difference
// between TensorOpIteration and TensorArgOpIteration, is that the latter store the index of the result, instead of 
// the result. The reason that they aren't combined is because of performance.
template <class ElemType, size_t N, int k>
struct TensorArgOpIteration
{
    static inline void Loop(array<ElemType*, N> pointers,
        const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrides,
        const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides, ElementWiseOperator reductionOp)
    {
        // non-scalar case: still nested result loops left
        array<ptrdiff_t, N> strides;
        for (size_t i = 0; i < N; i++) // N = a small constant, this will be unrolled
            strides[i] = regularStrides[i][(size_t)k];
        for (size_t dim = regularOpDims[(size_t)k]; dim-- > 0;)
        {
            // need to descend into one loop deeper
            TensorArgOpIteration<ElemType, N, k - 1>::Loop(pointers, regularOpDims, regularStrides, reducingOpDims, reducingStrides, reductionOp);
            // advance the pointers
            for (size_t i = 0; i < N; i++)
                pointers[i] += strides[i];
        }
    }
};

template <class ElemType, size_t N>
struct TensorArgOpIteration<ElemType, N, -1>
{
    static inline void Loop(array<ElemType*, N> pointers,
        const SmallVector<size_t>&, const array<SmallVector<ptrdiff_t>, N>&,
        const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides, ElementWiseOperator reductionOp)
    {
        // we are at element level for the result: perform the op (there may still be reduction)
        auto val = TensorArgOpReduction<ElemType, N, 2>::ReduceAll(pointers, reducingOpDims, reducingStrides, reductionOp);

        auto* pout = pointers.back();
        *pout = (ElemType)val.second;
        return;
    }
};

// -----------------------------------------------------------------------
// map runtime parameters N to template parameters
// -----------------------------------------------------------------------

// tensor operation with k+1 dimensions (-1 means scalar)
template <class ElemType, typename OPFN, typename ReductionOp, size_t N, int k>
static void TensorOpWithRegularLoop(ElemType beta, const array<ElemType*, N>& pointers, ElemType alpha, const OPFN& opfn, ReductionOp reductionOp,
                                    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrides,
                                    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
{
    size_t dims = reducingOpDims.size();
    switch (dims)
    {
    case 2:
        return TensorOpIteration<ElemType, OPFN, ReductionOp, N, false /*vectorizable*/, 1, k>::Loop(beta, pointers, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 1:
        return TensorOpIteration<ElemType, OPFN, ReductionOp, N, false /*vectorizable*/, 0, k>::Loop(beta, pointers, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 0:
    {
        // if all leading dimensions are 1, we can let the compiler do some unrolling
        bool leadingAllOne = true;
        for (size_t i = 0; i < N; i++)
            leadingAllOne &= k >= 0 && regularStrides[i][0] == 1;
        if (leadingAllOne) // special version that uses a hard-coded increment of 1 for all leading dimensions
            return TensorOpIteration<ElemType, OPFN, ReductionOp, N, true /*vectorizable*/, -1, k>::Loop(beta, pointers, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
        else
            return TensorOpIteration<ElemType, OPFN, ReductionOp, N, false /*vectorizable*/, -1, k>::Loop(beta, pointers, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    }
    default:
        LogicError("TensorOp: %d non-flattened reduction dimensions are not supported.", (int) dims);
    }
}

// tensor operation, generalized in number of arguments, operation already provided as a lambda
// This function now expands into different k.
template <class ElemType, typename OPFN, typename ReductionOp, size_t N>
static void TensorOpWithFnAndReduction(ElemType beta, array<ElemType*, N> pointers, ElemType alpha, const OPFN& opfn, const ReductionOp& reductionOp,
    const array<size_t, N>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
{
    for (size_t i = 0; i < N; i++) // N = a small constant, this will be unrolled
        pointers[i] += offsets[i];
    size_t dims = regularOpDims.size();
    switch (dims)
    {
    // N.B. consider code size impact when adding more cases.
    case 5:
        return TensorOpWithRegularLoop<ElemType, OPFN, ReductionOp, N, 4>(beta, pointers, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 4:
        return TensorOpWithRegularLoop<ElemType, OPFN, ReductionOp, N, 3>(beta, pointers, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 3:
        return TensorOpWithRegularLoop<ElemType, OPFN, ReductionOp, N, 2>(beta, pointers, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 2:
        return TensorOpWithRegularLoop<ElemType, OPFN, ReductionOp, N, 1>(beta, pointers, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 1:
        return TensorOpWithRegularLoop<ElemType, OPFN, ReductionOp, N, 0>(beta, pointers, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    case 0:
        return TensorOpWithRegularLoop<ElemType, OPFN, ReductionOp, N, -1>(beta, pointers, alpha, opfn, reductionOp, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
    default:
        LogicError("TensorOp: %d non-flattened input dimensions are not supported.", (int)dims);
    }
}

// tensor operation, generalized in number of arguments, operation already provided as a lambda
// This function now expands into different reductionOps
template <class ElemType, typename OPFN, size_t N>
static void TensorOpWithFn(ElemType beta, array<ElemType*, N> pointers, ElemType alpha, const OPFN& opfn, ElementWiseOperator reductionOp,
    const array<size_t, N>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides)
{
// BUGBUG: Using always 'double' as type of aggregator even for ElemType==float. Reason: otherwise some e2e test would fail as historically we 
// used double for aggregator of sum. But:
// * for min and max reductions this is meaningless.
// * It is not consitent with what we do on GPU, there we aggregate on ElemType.
// * It costs performance.
// TODO: apdapt e2e tests to run with aggregator of type ElemType.
#define CaseTensorOpWithFnAndReduction(oper)                                                  \
    case ElementWiseOperator::op##oper:                                                       \
    return TensorOpWithFnAndReduction(beta, pointers, alpha, opfn, [](double a, double b)     \
                                    {                                                         \
                                    return Op##oper(a, b);                                    \
                                    },                                                        \
                                    offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides)

    switch (reductionOp)
    {
        CaseTensorOpWithFnAndReduction(Sum);
        CaseTensorOpWithFnAndReduction(LogSum);
        CaseTensorOpWithFnAndReduction(Min);
        CaseTensorOpWithFnAndReduction(Max);
        CaseTensorOpWithFnAndReduction(ElementwiseProduct);
    default:
        LogicError("Specified ElementWiseOperator op %d not supported as reduction operation.", (int)reductionOp);
    }
}

// -----------------------------------------------------------------------
// entry points from Matrix.cpp; also map op to a lambda
// -----------------------------------------------------------------------

// special tensor ops for inference speed
template <class ElemType>
bool CPUMatrixSpecialUnaryTensorOpImpl(ElemType beta, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& o, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 2>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides);

template <class ElemType>
bool CPUMatrixSpecialBinaryTensorOpImpl(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& o, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 3>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides);

template <class ElemType>
bool CPUMatrixSpecialTernaryTensorOpImpl(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& c, CPUMatrix<ElemType>& o, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
    const array<size_t, 4>& offsets,
    const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
    const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides);

// perform unary operation 'op' on a giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
// This maps 'op' to a lambda.
template <class ElemType>
void CPUMatrixTensorOpImpl(ElemType beta, const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& o, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                           const array<size_t, 2>& offsets,
                           const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                           const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
{
    if (reductionOp != ElementWiseOperator::opSum    &&
        reductionOp != ElementWiseOperator::opLogSum &&
        reductionOp != ElementWiseOperator::opMin    &&
        reductionOp != ElementWiseOperator::opMax    &&
        reductionOp != ElementWiseOperator::opElementwiseProduct)
        InvalidArgument("TensorOp: Unary reduction operations other than opMax, opMin, opSum, and opLogSum are not implemented.");

#ifdef USE_MKL
    if (!!(CPUMatrix<ElemType>::GetOptimizationFlags() & CPUMatrix<ElemType>::OPT_EVAL_WITH_MKL) &&
        CPUMatrixSpecialUnaryTensorOpImpl(beta, a, o, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides))
        return;
#endif

// TODO: Change the lambda to take a pointer and a number of elements, so that we can pass it 1 or 4 elements, in order for it to SSE-vectorize.
#define CaseUnaryTensorOp(oper)                                                        \
    case ElementWiseOperator::op##oper:                                                \
        return TensorOpWithFn(beta, pointers, alpha, [](const array<ElemType*, 2>& pp) \
                              {                                                        \
                                  return Op##oper((*(pp[0])));                         \
                              },                                                       \
                              reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides)

    array<ElemType*, 2> pointers = {a.Data(), o.Data()};
    switch (op)
    {
        ForAllUnaryOps(CaseUnaryTensorOp);
    default:
        LogicError("TensorOp: Unknown unary op code %d.", (int) op);
    }
}

// perform binary operation 'op' on a and b giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
// This maps 'op' to a lambda.
template <class ElemType>
void CPUMatrixTensorOpImpl(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, CPUMatrix<ElemType>& o, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                           const array<size_t, 3>& offsets,
                           const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
                           const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides)
{
    if (reductionOp != ElementWiseOperator::opSum)
        InvalidArgument("TensorOp (binary): The only permitted binary reduction operation is opSum.");

#ifdef USE_MKL
    if (!!(CPUMatrix<ElemType>::GetOptimizationFlags() & CPUMatrix<ElemType>::OPT_EVAL_WITH_MKL) &&
        CPUMatrixSpecialBinaryTensorOpImpl(beta, a, b, o, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides))
        return;
#endif

#define CaseBinaryTensorOp(oper)                                                       \
    case ElementWiseOperator::op##oper:                                                \
        return TensorOpWithFn(beta, pointers, alpha, [](const array<ElemType*, 3>& pp) \
                              {                                                        \
                                  return Op##oper((*(pp[0])), (*(pp[1])));             \
                              },                                                       \
                              reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides)

    array<ElemType*, 3> pointers = {a.Data(), b.Data(), o.Data()};
    switch (op)
    {
        ForAllBinaryOps(CaseBinaryTensorOp);
    default:
        LogicError("TensorOp: Unknown op binary code %d.", (int) op);
    }
}

// perform ternary operation 'op' on a, and c giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
// This maps 'op' to a lambda.
template <class ElemType>
void CPUMatrixTensorOpImpl(ElemType beta, const CPUMatrix<ElemType>& a, const CPUMatrix<ElemType>& b, const CPUMatrix<ElemType>& c, CPUMatrix<ElemType>& o, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                           const array<size_t, 4>& offsets,
                           const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
                           const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides)
{
    if (reductionOp != ElementWiseOperator::opSum)
        InvalidArgument("TensorOp: The only permitted ternary reduction operation is opSum.");

#ifdef USE_MKL
    if (!!(CPUMatrix<ElemType>::GetOptimizationFlags() & CPUMatrix<ElemType>::OPT_EVAL_WITH_MKL) &&
        CPUMatrixSpecialTernaryTensorOpImpl(beta, a, b, c, o, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides))
        return;
#endif

#define CaseTernaryTensorOp(oper)                                                      \
    case ElementWiseOperator::op##oper:                                                \
        return TensorOpWithFn(beta, pointers, alpha, [](const array<ElemType*, 4>& pp) \
                              {                                                        \
                                  return Op##oper((*(pp[0])), (*(pp[1])), (*(pp[2]))); \
                              },                                                       \
                              reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides)

    array<ElemType*, 4> pointers = {a.Data(), b.Data(), c.Data(), o.Data()};
    switch (op)
    {
        ForAllTernaryOps(CaseTernaryTensorOp);
    default:
        LogicError("TensorOp: Unknown ternary op code %d.", (int) op);
    }
}

template <class ElemType>
void CPUMatrixTensorArgOpImpl(const CPUMatrix<ElemType>& a, CPUMatrix<ElemType>& o, ElementWiseOperator reductionOp,
                              const array<size_t, 2>& offsets,
                              const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                              const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
{
    if (reductionOp != ElementWiseOperator::opArgmin &&
        reductionOp != ElementWiseOperator::opArgmax)
        InvalidArgument("TensorOp: Arg reduction operations other than opArgmax, and opArgmin are not implemented.");

    if (o.GetNumElements() == 1)
    {
        o.Data()[0] = (ElemType) a.ArgOp(reductionOp);
    }
    else
    {
        const size_t N = 2;
        array<ElemType*, N> pointers = { a.Data(), o.Data() };
        for (size_t i = 0; i < N; i++)
            pointers[i] += offsets[i];

        switch (regularOpDims.size())
        {
            case 2:
                TensorArgOpIteration<ElemType, N, 1>::Loop(pointers, regularOpDims, regularStrides, reducingOpDims, reducingStrides, reductionOp);
                break;
            case 1:
                TensorArgOpIteration<ElemType, N, 0>::Loop(pointers, regularOpDims, regularStrides, reducingOpDims, reducingStrides, reductionOp);
                break;
            case 0:
                TensorArgOpIteration<ElemType, N, -1>::Loop(pointers, regularOpDims, regularStrides, reducingOpDims, reducingStrides, reductionOp);
                break;
            default:
                LogicError("TensorOp: %d non-flattened input dimensions are not supported.", (int)regularOpDims.size());
        }
    }
}

}}}