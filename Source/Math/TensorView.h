//
// <copyright file="TensorView.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

// This implements the TensorView class, which is a layer around Matrix that reinterprets its content as a generic tensor.

#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "DataTensor.h"

#pragma warning (push)
#pragma warning (disable: 4251) // needs to have dll-interface to be used by clients of... caused by TensorView::m_shape which is only private. We use the same compiler everywhere.

// This class is exported from the Math.dll.
namespace Microsoft { namespace MSR { namespace CNTK {

    template<class ElemType>
    class MATH_API TensorView
    {
    public:
        // -------------------------------------------------------------------
        // construction
        // -------------------------------------------------------------------

        // cast a matrix storage object (SOB) as a TensorView (without shape change)
        TensorView(Matrix<ElemType> & sob);
        // reshape a TensorView
        TensorView(const TensorView<ElemType> & sob, const TensorShape & shape);
        // reinterpret a SOB as a TensorView with a given TensorShape
        TensorView(Matrix<ElemType> & sob, const TensorShape & shape) :
            TensorView(TensorView(sob)/*cast as a TensorView*/, shape/*with a shape*/)
        { }
        // copy constructor
        TensorView(const TensorView<ElemType> & other) :
            TensorView(other.m_sob, other.m_shape)
        { }
        // assignment is forbidden since we contain a reference
        // If you ever need this, change the reference to a pointer.
        void operator=(const TensorView & other) = delete;  // since we have a reference

        // -------------------------------------------------------------------
        // accessors
        // -------------------------------------------------------------------

        const Matrix<ElemType> & GetSOB() const { return m_sob; }
        const TensorShape & GetShape() const { return m_shape; }

        // -------------------------------------------------------------------
        // elementwise operations
        // Result goes into 'this', and can optionally be added to the existing value.
        // E.g. c.DoSumOf(beta,a,b,alpha) means c := beta * c + alpha * (a + b).
        //  and c.DoDiffOf(0, c, a, 1) means c -= a.
        // All operators support elementwise in-place operations, i.e. a, b, and c
        // may all reference the same underlying SOB.
        // If beta == 0, c is not read out, i.e. it can be uninitialized or contain NaNs.
        // -------------------------------------------------------------------

        void DoSumOf(ElemType beta, const TensorView & a, const TensorView & b, ElemType alpha) { DoBinaryOpOf(beta, a, b, alpha, 0); }

        static void Test();

    private:

        void DoBinaryOpOf(ElemType beta, const TensorView & a, const TensorView & b, ElemType alpha, int op/*will become an enum later*/);

        // -------------------------------------------------------------------
        // sob members
        // -------------------------------------------------------------------

        Matrix<ElemType> & m_sob; // Storage OBject that holds the data that is being viewed with this TensorView
        TensorShape m_shape;            // the meta-data that describes the data's shape and/or access pattern
        // TODO: use a reference here or not? With a reference, we can hide more info in here such as cuDNN handles
    };

}}}

#pragma warning (pop)
