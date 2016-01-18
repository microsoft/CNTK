// TensorView.cpp -- main CPP file that contains all functions exported by the CNTKMath.dll
//
// <copyright file="Matrix.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

// TODO:
//  - dimension inference in nodes
//  - reduction on GPU is highly inefficient; test cases Image/QuickE2E PlusNode::BackpropTo() and ScaleNode::BackpropTo()
//  - accuracy deviation in FullUtterance and SequenceTraining
//  - TimesNode  --needs to specify reduction dimensions
//  - ConvolutionNode   --needs to specify reduction dimensions
//  - some nodes create new "dimensions" such as RowStack. Should that be an actual new tensor dimension?

// This implements the TensorView class, which is a layer around Matrix that reinterprets its content as a generic tensor.

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

// cast a matrix as a TensorView
template <class ElemType>
TensorView<ElemType>::TensorView(const Matrix<ElemType>& sob)
    : m_sob(sob.AsReference()), m_shape(TensorShape(array<size_t, 2>{sob.GetNumRows(), sob.GetNumCols()}))
{
}
// reshape a TensorView
template <class ElemType>
TensorView<ElemType>::TensorView(const TensorView<ElemType>& other, const TensorShape& shape)
    : m_sob(other.m_sob.AsReference()), m_shape(shape)
{
#if 0 // disabled since now we use slices, for which this check is no longer correct
        // for now we enforce that tensor dimensions match dimensions of the underlying matrix storage object
        // This is for sanity checks. In the future, it may appropriate to reduce this check to just checking the total number of elements, to allow abuses.
        // TODO: Use the multipliers instead?
        size_t i;
        size_t rowDim = 1;
        for (i = 0; i < m_shape.size() && rowDim < m_sob.GetNumRows(); i++)
            rowDim *= m_shape[i];
        // first i dimensions match matrix row dimension
        size_t colDim = 1;
        for (; i < m_shape.size(); i++)
            colDim *= m_shape[i];
        if (rowDim != m_sob.GetNumRows() || colDim != m_sob.GetNumCols())
            LogicError("TensorView: Tensor dimensions %s do not match storage-object dims %d x %d", string(m_shape).c_str(), (int)m_sob.GetNumRows(), (int)m_sob.GetNumCols());
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
            shapes[i].PadInPlace(dims);
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
    //fprintf(stderr, "Pre-flatten: Op %d: %s op %s -> %s via %s\n", (int)op, string(shapes[0]).c_str(), string(shapes[1]).c_str(), string(shapes[2]).c_str(), string(TensorShape(opDims)).c_str());
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
    //fprintf(stderr, "Post-flatten: Op %d: %s op %s -> %s via %s\n", (int)op, string(shapes[0]).c_str(), string(shapes[1]).c_str(), string(shapes[2]).c_str(), string(TensorShape(opDims)).c_str());

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
    //fprintf(stderr, "Post-drop: Op %d: %s op %s -> %s via %s\n", (int)op, string(shapes[0]).c_str(), string(shapes[1]).c_str(), string(shapes[2]).c_str(), string(TensorShape(opDims)).c_str());

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

    //fprintf(stderr, "%s  op  %s  ->  %s  via  %s\n", string(shapes[0]).c_str(), string(shapes[1]).c_str(), string(shapes[2]).c_str(), string(TensorShape(opDims)).c_str());

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
    //static int cc = 0; if (cc++ == 0)
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
    //static int cc = 0; if (cc++ == 0)
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
    //static int cc = 0; if (cc++ == 0)
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

template class TensorView<float>;
template class TensorView<double>;
} } }
