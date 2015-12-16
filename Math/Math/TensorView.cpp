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
    TensorView<ElemType>::TensorView(Matrix<ElemType> & sob) :
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

    static bool Matches(size_t d1, size_t d2) { return d1 == 1 || d2 == 1 || d1 == d2; }    // do two dimensions match?

    template<class ElemType>
    void TensorView<ElemType>::DoBinaryOpOf(ElemType beta, const TensorView & a, const TensorView & b, ElemType alpha, int op/*will become an enum later*/)
    {
        TensorView & c = *this;

        // TODO: an int matrix type would come in handy now... We can also use a vector<vector>.

        // massage TensorShapes
        // Note that TensorShapes here may be shapes are stored or shapes with stride magic applied.
        auto as = a.GetShape().GetDims();
        auto bs = b.GetShape().GetDims();
        auto cs = c.GetShape().GetDims();

        // expand ones to make tensors compatible
        // Trailing dimensions broadcast.
        // E.g. A(J) vs. B(J x T) will broadcast A(:) to all T columns.
        // To broadcast an A(T) to all J rows of B, use TensorShape editing to insert a dimension to get A(1,T).
        auto dims = max(max(as.size(), bs.size()), cs.size());
        as.resize(dims, 1);
        bs.resize(dims, 1);
        cs.resize(dims, 1);

        // determine operation shape (max over all dimensions)
        decltype(as) os(dims);
        for (size_t k = 0; k < dims; k++)
            os[k] = max(max(as[k], bs[k]), cs[k]);

        // dimension compatibility check
        // Each participant can broadcast. Non-broadcasting dimensions must match the operation dimension.
        for (size_t k = 0; k < dims; k++)
        {
            if (!Matches(as[k], os[k]) || !Matches(bs[k], os[k]) || !Matches(cs[k], os[k]))
                InvalidArgument("Binary tensor operation: Dimension %d is incompatible between the two inputs and output (%d vs. %d vs. %d)", (int)as[k], (int)bs[k], (int)cs[k]);
        }

        // flatten consecutive dimensions
        // Dimensions must be consecutive in memory, and either non-broadcasting or all-broadcasting, across all dimensions.
        // After this, as, bs, and cs no longer match the TensorShape objects.
        for (size_t k = 1; k < dims; k++)
        {
            // check if stored without gaps to skip
            if (!a.GetShape().CanFlatten(k) || !b.GetShape().CanFlatten(k) || !c.GetShape().CanFlatten(k))
                continue;
            // check if they are either all broadcasting or all not broadcasting
            if ((as[k] != os[k] || as[k - 1] != os[k - 1]) && (as[k] != 1 || as[k - 1] != 1))
                continue;
            if ((bs[k] != os[k] || bs[k - 1] != os[k - 1]) && (bs[k] != 1 || bs[k - 1] != 1))
                continue;
            if ((cs[k] != os[k] || cs[k - 1] != os[k - 1]) && (cs[k] != 1 || cs[k - 1] != 1))
                continue;
            // merge the dimensions
            as[k] *= as[k - 1]; as[k - 1] = 1;
            bs[k] *= bs[k - 1]; bs[k - 1] = 1;
            cs[k] *= cs[k - 1]; cs[k - 1] = 1;
            // BUGBUG: Must update multipliers as well
        }

        // remove singleton dimensions
        size_t j = 0;
        for (size_t k = 0; k < dims; k++)
        {
            if (as[k] == 1 && bs[k] == 1 && cs[k] == 1) // skip all-singleton dimensions
                continue;
            as[j] = as[k];
            bs[j] = bs[k];
            cs[j] = cs[k];
            os[j] = os[k];
            j++;
        }
        // note: if op is a scalar, then we end up with 0 dimensions here
        dims = j;
        as.resize(dims);
        bs.resize(dims);
        cs.resize(dims);
        os.resize(dims);
        let as1 = TensorShape(as);   // BUGBUG: We just lost stride info.
        let bs1 = TensorShape(bs);
        let cs1 = TensorShape(cs);
        let os1 = TensorShape(os);

        // determine inverse broadcasting dimensions
        // TODO: describe the resulting for loop as a set of tensor dims and strides as well.
        vector<bool> cBroadcasts(dims);
        for (size_t k = 0; k < dims; k++)
            cBroadcasts[k] = cs1[k] == 1 && (as1[k] != 1 || bs1[k] != 1);

        // now perform the operation
        fprintf(stderr, "Op %d: %s op %s -> %s via %s\n", (int)op, string(as1).c_str(), string(bs1).c_str(), string(cs1).c_str(), string(os1).c_str());
        // :)
        beta; alpha;
    }

    // simple test function for testing stuff
    // Call as: Microsoft::MSR::CNTK::TensorView<float>::Test();
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
        auto t3 = TensorView<ElemType>(m3, s3); t3;
        t3.DoSumOf(0, t1, t2, 1);
    }

    template class TensorView<float>;
    template class TensorView<double>;

}}}
