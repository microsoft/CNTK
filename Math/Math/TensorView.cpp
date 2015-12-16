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

namespace Microsoft {
    namespace MSR {
        namespace CNTK {

    using namespace std;

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

    // simple test function for testing stuff
    template<class ElemType>
    /*static*/ void TensorView<ElemType>::Test()
    {
        Matrix<ElemType> m(0);
        m.Resize(13, 42);
        TensorShape s1(13, 2, 22);
        TensorShape s2(13, 2, 21);
        s1;//let t1 = TensorView<ElemType>(m, s1); t1;
        let t2 = TensorView<ElemType>(m, s2); t2;
    }

    template class TensorView<float>;
    template class TensorView<double>;

}}}
