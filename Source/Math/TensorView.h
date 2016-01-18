//
// <copyright file="TensorView.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

// This implements the TensorView class, which is a layer around Matrix that reinterprets its content as a generic tensor. [fseide]

#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "TensorShape.h"

#pragma warning(push)
#pragma warning(disable : 4251) // needs to have dll-interface to be used by clients of... caused by TensorView::m_shape which is only private. We use the same compiler everywhere.

// This class is exported from the Math.dll.
namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class MATH_API TensorView
{
public:
    // -------------------------------------------------------------------
    // construction
    // -------------------------------------------------------------------

    // cast a matrix storage object (SOB) as a TensorView (without shape change)
    TensorView(const Matrix<ElemType>& sob);
    // reshape a TensorView
    TensorView(const TensorView<ElemType>& other, const TensorShape& shape);
    // reinterpret a SOB as a TensorView with a given TensorShape
    TensorView(const Matrix<ElemType>& sob, const TensorShape& shape)
        : TensorView(TensorView(sob) /*cast as a TensorView*/, shape /*with a shape*/)
    {
    }
    // empty constructor
    TensorView()
    {
    }
    // copy constructor
    TensorView(const TensorView<ElemType>& other)
        : m_sob(other.m_sob.AsReference()), m_shape(other.m_shape)
    {
    }

// -------------------------------------------------------------------
// elementwise operations
// Result goes into 'this', and can optionally be added to the existing value.
// E.g. c.DoSumOf(beta,a,b,alpha) means c := beta * c + alpha * (a + b),
//      c.AssignDiffOf(c,a) means c -= a,
//  and c.AddElementwiseProductOf(a, b, 1) means c += a .* b.
// All operators support elementwise in-place operations, i.e. a, b, and c
// may all reference the same underlying SOB, with onee exception:
// The output cannot be in-place and inverse-broadcasting at the same time.
// E.g. with c=[10] and a=[10 x 20], c.AssignDiffOf(c,a) will fail.
// In that case, you can use c.AddCopyOf(a,-1).
// Aliasing is not detected, so don't pass distinct TensorView objects that
// reference overlapping but not identical slices.
// If beta == 0, c is not read out, i.e. it can be uninitialized or contain NaNs.
// -------------------------------------------------------------------

#pragma push_macro("DeclareUnaryTensorOp")
#define DeclareUnaryTensorOp(oper)                                        \
    void Do##oper##Of(ElemType beta, const TensorView& a, ElemType alpha) \
    {                                                                     \
        DoUnaryOpOf(beta, a, alpha, ElementWiseOperator::op##oper);       \
    }                                                                     \
    void Assign##oper##Of(const TensorView& a, ElemType alpha = 1.0f)     \
    {                                                                     \
        DoUnaryOpOf(0, a, alpha, ElementWiseOperator::op##oper);          \
    }                                                                     \
    void Add##oper##Of(const TensorView& a, ElemType alpha = 1.0f)        \
    {                                                                     \
        DoUnaryOpOf(1.0f, a, alpha, ElementWiseOperator::op##oper);       \
    }

    ForAllUnaryOps(DeclareUnaryTensorOp);
#pragma pop_macro("DeclareUnaryTensorOp")

#pragma push_macro("DeclareBinaryTensorOp")
#define DeclareBinaryTensorOp(oper)                                                            \
    void Do##oper##Of(ElemType beta, const TensorView& a, const TensorView& b, ElemType alpha) \
    {                                                                                          \
        DoBinaryOpOf(beta, a, b, alpha, ElementWiseOperator::op##oper);                        \
    }                                                                                          \
    void Assign##oper##Of(const TensorView& a, const TensorView& b, ElemType alpha = 1.0f)     \
    {                                                                                          \
        DoBinaryOpOf(0, a, b, alpha, ElementWiseOperator::op##oper);                           \
    }                                                                                          \
    void Add##oper##Of(const TensorView& a, const TensorView& b, ElemType alpha = 1.0f)        \
    {                                                                                          \
        DoBinaryOpOf(1.0f, a, b, alpha, ElementWiseOperator::op##oper);                        \
    }

    ForAllBinaryOps(DeclareBinaryTensorOp);
#pragma pop_macro("DeclareBinaryTensorOp")

#pragma push_macro("DeclareTernaryTensorOp")
#define DeclareTernaryTensorOp(oper)                                                                                \
    void Do##oper##Of(ElemType beta, const TensorView& a, const TensorView& b, const TensorView& c, ElemType alpha) \
    {                                                                                                               \
        DoTernaryOpOf(beta, a, b, c, alpha, ElementWiseOperator::op##oper);                                         \
    }                                                                                                               \
    void Assign##oper##Of(const TensorView& a, const TensorView& b, const TensorView& c, ElemType alpha = 1.0f)     \
    {                                                                                                               \
        DoTernaryOpOf(0, a, b, c, alpha, ElementWiseOperator::op##oper);                                            \
    }                                                                                                               \
    void Add##oper##Of(const TensorView& a, const TensorView& b, const TensorView& c, ElemType alpha = 1.0f)        \
    {                                                                                                               \
        DoTernaryOpOf(1.0f, a, b, c, alpha, ElementWiseOperator::op##oper);                                         \
    }

    ForAllTernaryOps(DeclareTernaryTensorOp);
#pragma pop_macro("DeclareTernaryTensorOp")

    static void Test();

    void DoUnaryOpOf(ElemType beta, const TensorView& a, ElemType alpha, ElementWiseOperator op);
    void DoBinaryOpOf(ElemType beta, const TensorView& a, const TensorView& b, ElemType alpha, ElementWiseOperator op);
    void DoTernaryOpOf(ElemType beta, const TensorView& a, const TensorView& b, const TensorView& c, ElemType alpha, ElementWiseOperator op);

private:
    // -------------------------------------------------------------------
    // accessors
    // -------------------------------------------------------------------

    const Matrix<ElemType>& GetSOB() const
    {
        return m_sob;
    }
    Matrix<ElemType>& GetSOB()
    {
        return m_sob;
    }
    const TensorShape& GetShape() const
    {
        return m_shape;
    }

    // -------------------------------------------------------------------
    // sob members
    // -------------------------------------------------------------------

    Matrix<ElemType> m_sob; // Storage OBject that holds the data that is being viewed with this TensorView. This is really a reference (not owing the buffer).
    TensorShape m_shape;    // the meta-data that describes the data's shape and/or access pattern
    // TODO: use a reference here or not? With a reference, we can hide more info in here such as cuDNN handles
};
}
}
}

#pragma warning(pop)
