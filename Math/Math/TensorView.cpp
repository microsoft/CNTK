// TensorView.cpp -- main CPP file that contains all functions exported by the CNTKMath.dll
//
// <copyright file="Matrix.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

// This implements the TensorView class, which is a layer around Matrix that reinterprets its content as a generic tensor.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

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

    // cast a matrix as a tensor
    template<class ElemType>
    TensorView<ElemType>::TensorView(const Matrix<ElemType> & sob) :
        m_sob(sob), m_shape(TensorShape(array<size_t, 2> { sob.GetNumRows(), sob.GetNumCols() }))
    { }
    template<class ElemType>
    TensorView<ElemType>::TensorView(const TensorView<ElemType> & other, const TensorShape & shape) :
        m_sob(other.m_sob), m_shape(shape)
    {
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
    }

    // -------------------------------------------------------------------
    // elementwise operations
    // -------------------------------------------------------------------

    template<class ElemType>
    void TensorView<ElemType>::DoBinaryOpOf(const TensorView & a, const TensorView & b, TensorView & c, int op/*will become an enum later*/)
    {
        TensorView & c = *this;

        // massage TensorShapes
        auto as = a.GetShape().GetDims();
        auto bs = b.GetShape().GetDims();
        auto cs = c.GetShape().GetDims();

        // expand ones to make tensors compatible
        // Trailing dimensions broadcast.
        // E.g. A(J) vs. B(J x T) will broadcast A(:) to all T columns.
        // To broadcast an A(T) to all J rows of B, use TensorShape editing to insert a dimension to get A(1,T).
        let dims = max(max(as.size(), bs.size()), cs.size());
        as.resize(dims, 1);
        bs.resize(dims, 1);
        cs.resize(dims, 1);

        // compatibility check
        // Each participant can broadcast. Non-broadcasting dimensions must match.
        for (size_t k = 0; k < dims; k++)
        {
            dim = as[k];
            if (dim == 1)
                dim = bs[k];
            else if (bs[k] != 1 && dim != bs[k])
                InvalidArgument("Binary tensor operation: Dimension %d is incompatible between the two inputs (%d vs. %d)", (int)dim, (int)bs[k]);
            else if (cs[k] != 1 && dim != 1 && dim != cs[k])
                InvalidArgument("Binary tensor operation: Dimension %d is incompatible between inputs and output (%d vs. %d)", (int)dim, (int)cs[k]);
        }

        // flatten consecutive dimensions
        // Dimensions must be consecutive in memory, and either non-broadcasting or all-broadcasting, across all dimensions.

        // determine inverse broadcasting dimensions
        // TODO: describe the resulting for loop as a set of tensor dims and strides as well.
        vector<bool> cBroadcasts(dims);
        for (size_t k = 0; k < dims; k++)
            cBroadcasts[k] = cs[k] != 1 && dim == 1;

        // now perform the operation
        // :)
    }

    // simple test function for testing stuff
    template<class ElemType>
    /*static*/ void TensorView<ElemType>::Test()
    {
        Matrix<ElemType> m1(0); m1.Resize(1, 42);
        Matrix<ElemType> m2(0); m2.Resize(13, 1);
        Matrix<ElemType> m3(0); m3.Resize(13, 21);
        TensorShape s1(1, 2, 21);
        TensorShape s2(13, 1);
        TensorShape s3(13, 1, 21);
        let t1 = TensorView<ElemType>(m1, s1); t1;
        let t2 = TensorView<ElemType>(m2, s2); t2;
        let t3 = TensorView<ElemType>(m3, s3); t3;
        Add(m1, m2, m3);
    }

    template class TensorView<float>;
    template class TensorView<double>;

}}}
