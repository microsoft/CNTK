//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// TensorView.cpp -- main CPP file that contains all functions exported by the CNTKMath.dll
//
//

// TODO:
//  - dimension inference in nodes
//  - reduction on GPU is highly inefficient; test cases Image/QuickE2E PlusNode::BackpropTo() and ScaleNode::BackpropTo()
//  - accuracy deviation in FullUtterance and SequenceTraining
//  - TimesNode  --needs to specify reduction dimensions
//  - ConvolutionNode   --needs to specify reduction dimensions
//  - some nodes create new "dimensions" such as RowStack. Should that be an actual new tensor dimension?

// This implements the TensorView class, which is a layer around Matrix that reinterprets its content as a generic tensor.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "stdafx.h"
#include "Basics.h"
#include "TensorView.h"
#include <array>

#ifndef let
#define let const auto
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

// -------------------------------------------------------------------
// construction
// -------------------------------------------------------------------

// main constructor (all constructors except the default one route through this)
template <class ElemType>
TensorView<ElemType>::TensorView(const Matrix<ElemType>& sob, const TensorShape& shape)
    : m_sob(sob.AsReference()), m_shape(shape)
{
#ifdef _DEBUG
    // check bounds of TensorShape against underlying storage object
    // This is useful to detect errors like passing a matrix from the wrong input.
    const auto r = shape.GetLocationRange();
    const auto n = m_sob.GetNumElements();
    if (r.first < 0 || (size_t)r.second > n)
        LogicError("TensorView: Shape bounds [%d,%d) exceed bounds of underlying storage object [0,%d).", (int) r.first, (int) r.second, (int) n);
#endif
}

// -------------------------------------------------------------------
// elementwise operations
// -------------------------------------------------------------------

static bool Matches(size_t d1, size_t d2)
{
    return d1 == 1 || d2 == 1 || d1 == d2;
} // do two dimensions match?

template <class ElemType, size_t N>
static void PrepareTensorOperands(array<TensorShape, N> shapes, array<size_t, N>& offsets,
                                  SmallVector<size_t>& regularOpDims,
                                  array<SmallVector<ptrdiff_t>, N>& regularStrides,
                                  SmallVector<size_t>& reducingOpDims,
                                  array<SmallVector<ptrdiff_t>, N>& reducingStrides)
{
    // massage TensorShapes
    // Note that TensorShapes here may be shapes are stored or shapes with stride magic applied.

    // expand ones to make tensors compatible
    // Trailing dimensions broadcast.
    // E.g. A(J) vs. B(J x T) will broadcast A(:) to all T columns.
    // To broadcast an A(T) to all J rows of B, use TensorShape editing to insert a dimension to get A(1,T).
    size_t dims = 0;
    for (size_t i = 0; i < N; i++)
        if (dims < shapes[i].GetRank())
            dims = shapes[i].GetRank();
    for (size_t i = 0; i < N; i++)
        if (shapes[i].GetRank() < dims)
            shapes[i].PadRankInPlace(dims);
    // all shapes[] now have the same rank

    // determine operation shape (max over all dimensions)
    SmallVector<size_t> opDims(shapes[0].GetDims());
    for (size_t k = 0; k < dims; k++)
        for (size_t i = 1; i < N; i++)
            opDims[k] = max(opDims[k], shapes[i][k]);

    // dimension compatibility check
    // Each participant can broadcast. Non-broadcasting dimensions must match the operation dimension.
    for (size_t k = 0; k < dims; k++)
        for (size_t i = 0; i < N; i++)
            if (!Matches(shapes[i][k], opDims[k]))
                InvalidArgument("Binary tensor operation: Dimension %d of input [%d] is incompatible with operation dimensions (%s vs. %s)", (int) k, (int) i, string(shapes[i]).c_str(), string(TensorShape(opDims)).c_str());

    // flatten consecutive dimensions
    // Dimensions must be consecutive in memory, and either non-broadcasting or all-broadcasting, across all dimensions.
    // After this, as, bs, and cs no longer match the TensorShape objects.
    // fprintf(stderr, "Pre-flatten: Op %d: %s op %s -> %s via %s\n", (int)op, string(shapes[0]).c_str(), string(shapes[1]).c_str(), string(shapes[2]).c_str(), string(TensorShape(opDims)).c_str());
    for (size_t k = 1; k < dims; k++)
    {
        for (size_t i = 0; i < N; i++)
        {
            // check if stored without gaps to skip
            if (!shapes[i].CanFlatten(k))
                goto nope;
            // check if they are either all broadcasting or all not broadcasting
            if ((shapes[i][k] != opDims[k] || shapes[i][k - 1] != opDims[k - 1]) && (shapes[i][k] != 1 || shapes[i][k - 1] != 1))
                goto nope;
        }
        // these dimensions can be merged
        for (size_t i = 0; i < N; i++)
            shapes[i].FlattenInPlace(k);                          // TODO: overdoing the immutable thingy much?
        opDims = TensorShape(opDims).FlattenInPlace(k).GetDims(); // (ugh)
    nope:;
    }
    // fprintf(stderr, "Post-flatten: Op %d: %s op %s -> %s via %s\n", (int)op, string(shapes[0]).c_str(), string(shapes[1]).c_str(), string(shapes[2]).c_str(), string(TensorShape(opDims)).c_str());

    // remove singleton dimensions
    SmallVector<bool> toDrop(dims, false);
    bool anyToDrop = false;
    for (size_t k = 0; k < dims; k++)
    {
        for (size_t i = 0; i < N; i++)
            if (shapes[i][k] != 1)
                goto neither;
        toDrop[k] = true; // found an all-singleton dimensions
        anyToDrop = true;
    neither:;
    }
    if (anyToDrop)
    {
        for (size_t i = 0; i < N; i++)
            shapes[i].DropDimsInPlace(toDrop);
        opDims = TensorShape(opDims).DropDimsInPlace(toDrop).GetDims(); // (ugh)
        dims = opDims.size();                                           // #dims has changed
    }
    for (size_t i = 0; i < N; i++)
        assert(dims == shapes[i].size());
    // note: if op is a scalar, then we end up with 0 dimensions here, which is allowed
    // fprintf(stderr, "Post-drop: Op %d: %s op %s -> %s via %s\n", (int)op, string(shapes[0]).c_str(), string(shapes[1]).c_str(), string(shapes[2]).c_str(), string(TensorShape(opDims)).c_str());

    // determine broadcasting; that is, set strides to 0 for 1-dimensions
    // To be more precise, we should only set actually broadcasting dimensions to 0.
    // But since dimensions that are 1 across all args are eliminated, any 1 must be some form of broadcasting.
    for (size_t i = 0; i < N; i++) // TODO: do we need to test output tensor here as well?
        for (size_t k = 0; k < dims; k++)
            if (shapes[i][k] < opDims[k])
            {
                shapes[i].SetBroadcastStrides();
                break;
            }

    // fprintf(stderr, "%s  op  %s  ->  %s  via  %s\n", string(shapes[0]).c_str(), string(shapes[1]).c_str(), string(shapes[2]).c_str(), string(TensorShape(opDims)).c_str());

    // determine inverse broadcasting dimensions
    // Inverse broadcasting dims are actual for loops in the kernel, whereas broadcasting input dims are handled by the thread index.
    // For regular input dims:
    //  - determine number of steps (product over opDims[.])
    //  - launch that many kernels
    //  - pass in:
    //     - total number of steps
    //     - strides for all inputs (with stride magic), separated by regular and inverse broadcasting dimensions
    //     - opDim (no stride magic allowed) for regular broadcasting dimensions
    //     - reverse broadcasting dimensions
    //     - opcodes for elementwise op and reduction op
    //  - in each kernel:
    //     - map thread index to dimensions (regular broadcasting ones)
    //     - for-loop over inverse broadcasting dimensions
    //        - map dimensions (including inverse broadcasting) for every input
    //        - perform op on the input values
    //        - accumulate
    //     - map dimensions (regular) for output
    //     - save result

    // separate out the inverse-broadcasting dimensions
    // Any singleton dimension in the result tensor is inverse-broadcasting, because there must be at least one non-1 dimension
    // in one of the inputs, otherwise the entire dimension would have been optimized away above.
    SmallVector<bool> isReducingDim(dims); // true for each inverse-broadcasting dimension
    bool isAnyReducingDim = false;
    for (size_t k = 0; k < dims; k++)
    {
        bool isRed = shapes.back()[k] == 1;
        isReducingDim[k] = isRed;
        isAnyReducingDim |= isRed;
    }

    // form the regular (non-inverse-broadcasting) dims
    if (isAnyReducingDim)
    {
        for (size_t i = 0; i < N; i++)
            regularStrides[i] = shapes[i].DropDims(isReducingDim).GetStrides();
        regularOpDims = TensorShape(opDims).DropDims(isReducingDim).GetDims(); // (ugh)

        // form the inverse-broadcasting dims
        SmallVector<bool> isRegularDim(dims); // true for each inverse-broadcasting dimension
        for (size_t k = 0; k < dims; k++)
            isRegularDim[k] = !isReducingDim[k]; // (no way to do this more nicely?)
        for (size_t i = 0; i < N; i++)
            reducingStrides[i] = shapes[i].DropDims(isRegularDim).GetStrides();
        reducingOpDims = TensorShape(opDims).DropDims(isRegularDim).GetDims(); // (ugh)
    }
    else // case if no reduction: things are simpler
    {
        for (size_t i = 0; i < N; i++)
            regularStrides[i] = shapes[i].GetStrides();
        regularOpDims = opDims;

        for (size_t i = 0; i < N; i++)
            reducingStrides[i].clear();
        reducingOpDims.clear();
    }

    for (size_t i = 0; i < N; i++)
        offsets[i] = shapes[i].GetOffset();
}

// enforce that in case of broadcasting, the output must not be an input
template <class ElemType>
static bool CheckDifferentObject(const TensorView<ElemType>& a, const TensorView<ElemType>& b)
{
    if (&a == &b)
        LogicError("Do{U,Bi,Ter}naryOpOf: When inverse broadcasting, output must not be an input.");
    return true;
}

template <class ElemType>
void TensorView<ElemType>::DoUnaryOpOf(ElemType beta, const TensorView& a, ElemType alpha, ElementWiseOperator op)
{
    // static int cc = 0; if (cc++ == 0)
    //    fprintf(stderr, "Tensor Op: Op %d: %s -> %s\n", (int)op, string(a.GetShape()).c_str(), string(GetShape()).c_str());

    // prepare all tensor descriptor information as needed for execution
    array<size_t, 2> offsets;
    array<SmallVector<ptrdiff_t>, 2> regularStrides, reducingStrides;
    SmallVector<size_t> regularOpDims, reducingOpDims;
    PrepareTensorOperands<ElemType, 2>(array<TensorShape, 2>{a.GetShape(), GetShape()}, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);

    // output cannot be input when reducing
    if (reducingOpDims.size() > 0)
        CheckDifferentObject(a, *this);

    // now perform the operation
    GetSOB().TensorOp(beta, a.GetSOB(), alpha, op, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
}

template <class ElemType>
void TensorView<ElemType>::DoBinaryOpOf(ElemType beta, const TensorView& a, const TensorView& b, ElemType alpha, ElementWiseOperator op)
{
    // static int cc = 0; if (cc++ == 0)
    //    fprintf(stderr, "Tensor Op: Op %d: %s op %s -> %s\n", (int)op, string(a.GetShape()).c_str(), string(b.GetShape()).c_str(), string(GetShape()).c_str());

    array<size_t, 3> offsets;
    array<SmallVector<ptrdiff_t>, 3> regularStrides, reducingStrides;
    SmallVector<size_t> regularOpDims, reducingOpDims;
    PrepareTensorOperands<ElemType, 3>(array<TensorShape, 3>{a.GetShape(), b.GetShape(), GetShape()}, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);

    // output cannot be input when reducing
    if (reducingOpDims.size() > 0)
        CheckDifferentObject(a, *this) && CheckDifferentObject(b, *this);

    GetSOB().TensorOp(beta, a.GetSOB(), b.GetSOB(), alpha, op, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
}

template <class ElemType>
void TensorView<ElemType>::DoTernaryOpOf(ElemType beta, const TensorView& a, const TensorView& b, const TensorView& c, ElemType alpha, ElementWiseOperator op)
{
    // static int cc = 0; if (cc++ == 0)
    //    fprintf(stderr, "Tensor Op: Op %d: %s, %s, %s -> %s\n", (int)op, string(a.GetShape()).c_str(), string(b.GetShape()).c_str(), string(c.GetShape()).c_str(), string(GetShape()).c_str());

    array<size_t, 4> offsets;
    array<SmallVector<ptrdiff_t>, 4> regularStrides, reducingStrides;
    SmallVector<size_t> regularOpDims, reducingOpDims;
    PrepareTensorOperands<ElemType, 4>(array<TensorShape, 4>{a.GetShape(), b.GetShape(), c.GetShape(), GetShape()}, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);

    // output cannot be input when reducing
    if (reducingOpDims.size() > 0)
        CheckDifferentObject(a, *this) && CheckDifferentObject(b, *this) && CheckDifferentObject(c, *this);

    GetSOB().TensorOp(beta, a.GetSOB(), b.GetSOB(), c.GetSOB(), alpha, op, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
}

// simple test function for testing stuff
// Call as: Microsoft::MSR::CNTK::TensorView<float>::Test();
template <class ElemType>
/*static*/ void TensorView<ElemType>::Test()
{
    const DEVICEID_TYPE deviceId = 0; // -1
    Matrix<ElemType> m1(deviceId);
    Matrix<ElemType> m2(deviceId);
    Matrix<ElemType> m3(deviceId);
    {
        m1.SetValue(5, 3, {1, 2, 3,
                           14, 15, 6,
                           4, 5, 16,
                           41, 5, 1,
                           1.8, 4.5, 7});
        m2.SetValue(5, 1, {42,
                           13,
                           1968,
                           3.1415f,
                           7});

        m3.Resize(m1);

        // regular zip  (just add m1 to itself)
        TensorView(m3).DoSumOf(0, TensorView(m1), TensorView(m1), 1);
        m3.Print();

        // unary op
        TensorView(m3).DoSqrtOf(0, TensorView(m1), 1);
        m3.Print();

        // broadcasting of an input
        TensorView(m3).DoSumOf(0, TensorView(m1), TensorView(m2), 1);
        m3.Print();

        TensorView(m3).DoMaxOf(0, TensorView(m1), TensorView(m2), 1);
        m3.Print();

        TensorView(m3).DoGTOf(0, TensorView(m1), TensorView(m2), 1);
        m3.Print();

        // reduction over columns
        m3.Resize(5, 1);
        TensorView(m3).DoSumOf(0, TensorView(m1), TensorView(m2), 1);
        m3.Print();

        // reduction over rows
        m3.Resize(1, 3);
        TensorView(m3).DoSumOf(0, TensorView(m1), TensorView(m2), 1);
        m3.Print();

        TensorView(m3).DoLogSumOf(0, TensorView(m1), TensorView(m2), 1);
        m3.Print();
    }
    {
        m1.Resize(1, 42);
        m2.Resize(13, 1);
        m3.Resize(13, 21);
        TensorShape s1(1, 2, 21);
        TensorShape s2(13, 1);
        TensorShape s3(13, 1, 21);
        let t1 = TensorView<ElemType>(m1, s1);
        t1;
        let t2 = TensorView<ElemType>(m2, s2);
        t2;
        auto t3 = TensorView<ElemType>(m3, s3);
        t3;
        t3.DoSumOf(0, t1, t2, 1);
        m3.Print();
    }
}

// -------------------------------------------------------------------
// matrix product -- GEMM for flattened tensors
// -------------------------------------------------------------------

// print the dimensions of a matrix-product operation, for pretty error reporting
static string MatrixProductFormat(const TensorShape& shapeA, bool transA, const TensorShape& shapeB, bool transB, const TensorShape& shapeC, bool transC)
{
    string result = "[" + string(shapeA) + "]"; if (transA) result.append("'");
    result += " * ";
    result +=       "[" + string(shapeB) + "]"; if (transB) result.append("'");
    result += " -> ";
    result +=       "[" + string(shapeC) + "]"; if (transC) result.append("'");
    return result;
}

// flatten a tensor into a 2D tensor, where splitPoint is the first index to go into the second dimension
// The tensor must be flattenable this way, i.e. each of the two index ranges must be dense.
static void FlattenToMatrix(TensorShape& shape, bool trans, size_t splitPoint)
{
    if (trans)
        splitPoint = shape.GetRank() - splitPoint;
    // check & print meaningful error message
    SmallVector<bool> dimsToDrop(shape.GetRank(), false);
    for (size_t k = 1; k < shape.GetRank(); k++)
        if (k != splitPoint)
            if (!shape.CanFlatten(k))
                InvalidArgument("DoMatrixProductOf: Shape [%s] is not dense at dimension %d.", string(shape).c_str(), (int)k);
            else
                dimsToDrop[k - 1] = true;
    // handle case where last dimension missing, e.g. u'v where u and v are column vectors
    if (splitPoint == shape.GetRank())
    {
        shape.PadRankInPlace(splitPoint + 1);
        dimsToDrop.resize(splitPoint + 1, false);
    }
    // flatten the dimensions
    for (size_t k = 1; k < shape.GetRank(); k++)
        if (dimsToDrop[k - 1])
            shape.FlattenInPlace(k);
    shape.DropDimsInPlace(dimsToDrop);
    // handle edge case where first dimension missing, e.g. u'v where both are scalars
    if (splitPoint == 0)
    {
        // we must insert a 1 dimension at the start
        assert(shape.GetRank() == 1); // we have reduced everything after the split point at this point
        shape.PadRankInPlace(2);      // append a 1
        shape.SwapDimsInPlace(0, 1);  // and swap--this preserves the stride of the second dimension
    }
    // now we have a matrix
    assert(shape.GetRank() == 2);
}

// convert tensor into a Matrix object
template <class ElemType>
Matrix/*ref*/<ElemType> TensorView<ElemType>::AsMatrix() const
{
    assert(m_shape.GetRank() == 2);
    if (m_shape.GetStrides()[0] != 1 && m_shape[0] != 1)
        InvalidArgument("AsMatrix: Flattened [%s] matrix is not dense (it has a stride).", string(m_shape).c_str());
    // create a Matrix view into the TensorView (which in turn is a view over a Matrix...)
    // The way to do this is to use a ColumnSlice.
    // express the TensorView's storage in m_sob's coordinates
    let firstColumn = m_shape.GetOffset()      / m_sob.GetNumRows();
    let numColumns  = m_shape.GetNumElements() / m_sob.GetNumRows();
    if (firstColumn * m_sob.GetNumRows() != m_shape.GetOffset() || numColumns * m_sob.GetNumRows() != m_shape.GetNumElements())
        InvalidArgument("AsMatrix: Flattened [%s] matrix has an offset or width that is not a multiple of the storage object's row dimension.", string(m_shape).c_str());
    auto sob = m_sob.ColumnSlice(firstColumn, numColumns);
    // now reinterpret this slice according to the new tensor shape
    // Example:
    //  - each sob column contains a set of vectors stored as a 2D tensor [I x J], and [S x T] samples
    //  - we want to apply a [K x I] matrix to all vectors in each set
    //  - so we reinterpret the [(I * J) x (S * T)] storage object as a [I x (J * S * T)] matrix
    //    and apply the matrix product to this (by calling GEMM)
    //  - which in turn yields a [K x (J * S x*T)] matrix
    //    which gets reinterpreted back as a [K x J x S x T] tensor
    // In the special case of sparse matrices, this split cannot be done. E.g. in the above example, we could only multiply with a [K x I x J] tensor.
    if (sob.GetMatrixType() == MatrixType::DENSE)
        return sob.Reshaped(m_shape[0], m_shape[1]);
    else if (m_shape[0] == sob.GetNumRows()) // SPARSE matrices cannot be reshaped, so we only support 1D and 2D tensors
        return sob;
    else
        RuntimeError("AsMatrix: Sparse tensors are not supported unless they are 1D or 2D matrices.");
}

template <class ElemType>
void TensorView<ElemType>::DoMatrixProductOf(ElemType beta, bool transC, const TensorView& a, bool transA, const TensorView& b, bool transB, ElemType alpha)
{
    // determine integration dimension offset
    auto shapeA = a.m_shape;
    auto shapeB = b.m_shape;
    auto shapeC =   m_shape;
    if (shapeA.GetRank() + shapeB.GetRank() < shapeC.GetRank())
        InvalidArgument("DoMatrixProductOf: Ranks %s don't match, output must have a non-reduced output dimension.", MatrixProductFormat(shapeA, transA, shapeB, transB, shapeC, transC).c_str());
    let removedDims = shapeA.GetRank() + shapeB.GetRank() - shapeC.GetRank();
    let numReducedDims = removedDims / 2;
    if (numReducedDims * 2 != removedDims)
        InvalidArgument("DoMatrixProductOf: Ranks %s mismatch.", MatrixProductFormat(shapeA, transA, shapeB, transB, shapeC, transC).c_str());
    let firstReducedDim = shapeA.GetRank() - numReducedDims;
    // flatten. This updates shapeA etc.
    FlattenToMatrix(shapeA, transA, firstReducedDim);
    FlattenToMatrix(shapeB, transB, numReducedDims);
    FlattenToMatrix(shapeC, transC, firstReducedDim);
    // check dimensions
    // shapeX[transX] and shapeX[1-transX] are row and column dim, respectively, or swapped if transposed
    if (shapeA[transA]   != shapeC[transC]   || // output dim
        shapeB[1-transB] != shapeC[1-transC] || // input dim
        shapeA[1-transA] != shapeB[transB])     // reduction dim
    {
        InvalidArgument("DoMatrixProductOf: Flattened tensor dimensions %s mismatch.", MatrixProductFormat(shapeA, transA, shapeB, transB, shapeC, transC).c_str());
    }
    // create Matrix objects out of this
    let  A = a.Reshaped(shapeA).AsMatrix();
    let  B = b.Reshaped(shapeB).AsMatrix();
    auto C =   Reshaped(shapeC).AsMatrix();
    // and go
    if (!transC)
        Matrix<ElemType>::MultiplyAndWeightedAdd(alpha, A, transA, B, transB, beta, C);
    else // C' = A * B  <==>  C = (A * B)' = B' * A'
        Matrix<ElemType>::MultiplyAndWeightedAdd(alpha, B, !transB, A, !transA, beta, C);
}

template class TensorView<float>;
template class TensorView<double>;

}}}
