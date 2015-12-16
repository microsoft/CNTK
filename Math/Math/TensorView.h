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

        // cast a matrix as a TensorView (without shape change)
        TensorView(const Matrix<ElemType> & sob);
        // reshape a TensorView
        TensorView(const TensorView<ElemType> & sob, const TensorShape & shape);
        // reinterpret a Matrix as a TensorView with reshaping
        TensorView(const Matrix<ElemType> & sob, const TensorShape & shape) :
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
        // operations
        // -------------------------------------------------------------------

        static void Test();

    private:

        // -------------------------------------------------------------------
        // sob members
        // -------------------------------------------------------------------

        const Matrix<ElemType> & m_sob; // Storage OBject that holds the data that is being viewed with this TensorView
        TensorShape m_shape;            // the meta-data that describes the data's shape and/or access pattern
    };

}}}

#pragma warning (pop)
